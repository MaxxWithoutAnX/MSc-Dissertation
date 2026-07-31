import csv
import io
import time
import torch
import torch.nn as nn
import numpy as np
from aurora import rollout
from eval_metrics import compute_all_metrics


def read_manifest(path):
    """Read a CSV manifest and return a list of row dicts."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def resume_skip(rows, existing_tags):
    """Manifest rows not already present in a partial results file."""
    return [r for r in rows if r["tag"] not in existing_tags]


def precision_groups(config):
    """Invert {group: precision} to {precision: [group, ...]}."""
    out = {}
    for g, p in config.items():
        out.setdefault(p, []).append(g)
    return out


def group_members(groups_map, group_names):
    members = set()
    for g in group_names:
        members.update(groups_map.get(g, []))
    return members


def torchao_config_for(precision):
    if precision == "bf16":
        return None
    if precision == "W8":
        from torchao.quantization.quant_api import Int8WeightOnlyConfig
        return Int8WeightOnlyConfig()
    if precision == "W8A8":
        from torchao.quantization.quant_api import Int8DynamicActivationInt8WeightConfig
        return Int8DynamicActivationInt8WeightConfig()
    if precision == "W4":
        from torchao.quantization.quant_api import UIntXWeightOnlyConfig
        return UIntXWeightOnlyConfig(dtype=torch.uint4, group_size=64)
    if precision == "W8A8_sq":
        raise NotImplementedError("SmoothQuant (W8A8_sq) is not implemented yet.")
    raise ValueError(f"unknown precision {precision!r}")


DEFAULT_IS_LINEAR = lambda m, fqn: (
    isinstance(m, nn.Linear) and m.weight.dim() == 2 and m.weight.shape[1] > 1)


def build_quantised_model(model, groups_map, config, is_linear=DEFAULT_IS_LINEAR):
    from torchao import quantize_
    by_precision = precision_groups(config)
    for precision in sorted(by_precision):
        if precision == "bf16":
            continue
        members = group_members(groups_map, by_precision[precision])
        cfg = torchao_config_for(precision)

        def filter_fn(mod, fqn, members=members):
            return is_linear(mod, fqn) and fqn in members

        quantize_(model, cfg, filter_fn=filter_fn)
    return model


def apply_dynamo_hardening():
    import torch._dynamo
    import torch._inductor.config as inductor_config
    torch._dynamo.config.cache_size_limit = 64
    for obj in (getattr(inductor_config, "triton", None), inductor_config):
        if obj is not None and hasattr(obj, "assert_indirect_indexing"):
            obj.assert_indirect_indexing = False


COMPILE_TARGETS = ("backbone", "encoder.level_agg", "decoder.level_decoder",
                   "decoder.level_decoder_alternate")


def _resolve_parent_attr(model, dotted):
    """(parent_obj, attr_name) for a dotted path, or (None, None)."""
    parts = dotted.split(".")
    obj = model
    for part in parts[:-1]:
        if not hasattr(obj, part):
            return None, None
        obj = getattr(obj, part)
    if not hasattr(obj, parts[-1]):
        return None, None
    return obj, parts[-1]


def compile_heavy_submodules(model, targets=COMPILE_TARGETS):
    compiled = []
    for dotted in targets:
        parent, name = _resolve_parent_attr(model, dotted)
        if parent is None:
            continue
        setattr(parent, name, torch.compile(getattr(parent, name)))
        compiled.append(dotted)
    return compiled


def model_size_gb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    size = buf.getbuffer().nbytes / 1e9
    buf.close()
    return size


def warmup_and_measure(rollout_once, steps, warmup_reps=1, timed_reps=3):
    for _ in range(warmup_reps):
        rollout_once()

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    rep_times = []
    for _ in range(timed_reps):
        if has_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        rollout_once()
        if has_cuda:
            torch.cuda.synchronize()
        rep_times.append(time.perf_counter() - t0)

    latency_s_per_step = float(np.median(rep_times)) / steps
    if has_cuda:
        vram_peak_gb = torch.cuda.max_memory_allocated() / 1e9
        vram_reserved_gb = torch.cuda.max_memory_reserved() / 1e9
    else:
        vram_peak_gb = None
        vram_reserved_gb = None
    return {"latency_s_per_step": latency_s_per_step,
            "vram_peak_gb": vram_peak_gb, "vram_reserved_gb": vram_reserved_gb}


def rollout_metrics_over_inits(model, init_indices, make_batch, steps, lead_by_step,
                                gt_lookup, variables, lat_eval, pred_to_tensor):
    out = {}
    for i in init_indices:
        out[i] = {}
        batch = make_batch(i)
        with torch.inference_mode():
            for k, pred in enumerate(rollout(model, batch, steps=steps)):
                if k in lead_by_step:
                    lead = lead_by_step[k]
                    gt = gt_lookup(i, lead)
                    if gt is None:
                        raise KeyError(f"no truth for init {i} at lead {lead}")
                    pred_t = pred_to_tensor(pred)
                    out[i][lead] = compute_all_metrics(pred_t, gt, None, variables,
                                                        lat_eval, lead)
                del pred
    return out


def measured_consistency_axes(fp32_run, config_run, tag, lead, dates, noise_floor=None):
    import ablation_comp as ac
    from distortion import NoiseFloor
    if noise_floor is None:
        noise_floor = NoiseFloor.from_detailed()
    specs = ac.available_specs(fp32_run, config_run, ac.metric_registry(fp32_run), lead)
    block = ac.bootstrap_block(len(fp32_run.dates))
    records = ac.sensitivity_records(fp32_run, {tag: config_run}, specs, [tag],
                                     [lead], dates, block, full_table=None,
                                     norm_mode="colmax", noise_floor=noise_floor)
    agg = ac.group_aggregates(records, specs, [tag], lead)[tag]
    return {"balance": agg["balance_distortion"], "conservation": agg["conservation_distortion"]}


def measured_consistency_axes_all_leads(fp32_run, config_run, tag, leads, dates, noise_floor=None):
    return {lead: measured_consistency_axes(fp32_run, config_run, tag, lead, dates, noise_floor)
            for lead in leads}


RESULTS_CSV_FIELDS = ["tag", "floor", "guide", "predicted_cost", "balance",
                      "conservation", "config", "model_size_gb",
                      "latency_s_per_step", "vram_peak_gb", "vram_reserved_gb",
                      "n_inits"]

RESULTS_BY_LEAD_CSV_FIELDS = ["tag", "floor", "guide", "lead", "balance", "conservation"]


def write_results_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in RESULTS_CSV_FIELDS})


def write_results_by_lead_csv(path, rows):
    """One row per (config, lead)."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_BY_LEAD_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in RESULTS_BY_LEAD_CSV_FIELDS})


def write_results_pt(path, per_config_metrics):
    torch.save(per_config_metrics, path)


def measure_config(tag, config, groups_map, build_model, make_batch, gt_lookup,
                   pred_to_tensor, variables, lat_eval, init_indices, steps,
                   lead_by_step, compile_enabled=True, warmup_reps=1, timed_reps=3,
                   is_linear=DEFAULT_IS_LINEAR):
    model = build_model()
    build_quantised_model(model, groups_map, config, is_linear=is_linear)
    if compile_enabled:
        apply_dynamo_hardening()
        compile_heavy_submodules(model)

    timing_init = init_indices[0]

    def rollout_once():
        batch = make_batch(timing_init)
        with torch.inference_mode():
            for _ in rollout(model, batch, steps=steps):
                pass

    stats = warmup_and_measure(rollout_once, steps, warmup_reps, timed_reps)
    stats["model_size_gb"] = model_size_gb(model)

    metrics = rollout_metrics_over_inits(model, init_indices, make_batch, steps,
                                        lead_by_step, gt_lookup, variables, lat_eval,
                                        pred_to_tensor)
    return metrics, stats
