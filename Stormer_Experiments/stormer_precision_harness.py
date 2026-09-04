""" Same as precision_harness.py in Aurora_Experiemnts but built for Stormer
"""
import csv
import io
import time

import numpy as np
import torch
import torch.nn as nn

import physq_path

from eval_metrics import compute_all_metrics, difference_kinetic_energy
from prepare_year_h5 import MODEL_VARIABLES


# --- manifest plumbing (verbatim from Aurora's precision_harness.py) -------------------

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


def _quantize_(model, cfg, filter_fn):
    """Indirection over torchao.quantize_ so calls can be observed without a GPU."""
    from torchao import quantize_
    quantize_(model, cfg, filter_fn=filter_fn)


def build_quantised_model(model, groups_map, config, is_linear=DEFAULT_IS_LINEAR):
    by_precision = precision_groups(config)
    for precision in sorted(by_precision):
        if precision == "bf16":
            continue
        members = group_members(groups_map, by_precision[precision])
        cfg = torchao_config_for(precision)

        def filter_fn(mod, fqn, members=members):
            return is_linear(mod, fqn) and fqn in members

        _quantize_(model, cfg, filter_fn)
    return model


# --- compile + measurement (verbatim, except COMPILE_TARGETS) -------------------------

def apply_dynamo_hardening():
    import torch._dynamo
    import torch._inductor.config as inductor_config
    torch._dynamo.config.cache_size_limit = 64
    for obj in (getattr(inductor_config, "triton", None), inductor_config):
        if obj is not None and hasattr(obj, "assert_indirect_indexing"):
            obj.assert_indirect_indexing = False


COMPILE_TARGETS = ("net",)


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
    """Wrap each present target submodule in torch.compile, in place."""
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


# --- CHANGE 1 of 4: Stormer's three-interval averaged rollout -------------------------

INTERVALS = (6, 12, 24)


def forwards_per_init(leads, intervals=INTERVALS):
    """Total forwards for one init: sum over intervals of max(leads)//interval."""
    return sum(max(leads) // inter for inter in intervals)


def stormer_rollout_to_leads(model, inp, vars_b, leads, intervals=INTERVALS):
    for inter in intervals:
        for lead in leads:
            assert lead % inter == 0, (
                f"lead {lead} is not divisible by interval {inter}; it would be "
                f"snapshotted in some interval rollouts and not others, averaging "
                f"unequal counts")

    max_lead = max(leads)
    preds_by_lead = {lead: [] for lead in leads}
    for inter in intervals:
        step_to_lead = {lead // inter: lead for lead in leads}
        interval_tensor = (torch.tensor([inter], device=inp.device, dtype=inp.dtype)
                           / 10.0).repeat(inp.shape[0])
        x = inp
        with torch.inference_mode():
            for step in range(1, max_lead // inter + 1):
                pred_diff = model(x, vars_b, interval_tensor)
                pred_diff = model.replace_constant(pred_diff, vars_b)
                pred_diff = model.reverse_diff_transform[inter](pred_diff)
                x = model.inp_transform(model.reverse_inp_transform(x) + pred_diff)
                if step in step_to_lead:
                    preds_by_lead[step_to_lead[step]].append(x)
    return {lead: torch.stack(preds_by_lead[lead], dim=0).mean(0) for lead in leads}


def stormer_metrics_over_inits(model, loader, leads, variables, lat,
                               reverse_inp_transform, init_key_of, ref_winds=None,
                               keep_winds=False):
    wind_vars = [v for v in variables
                 if v.startswith(("u_component_of_wind_", "v_component_of_wind_"))]
    wind_idx = [variables.index(v) for v in wind_vars]
    device = next(model.parameters()).device

    out, winds = {}, ({} if keep_winds else None)
    for i, (inp, gt_by_lead, vars_b) in enumerate(loader):
        key = init_key_of(i)
        preds = stormer_rollout_to_leads(model, inp.to(device), vars_b, leads)

        out[key] = {}
        if keep_winds:
            winds[key] = {}
        for lead in leads:
            pred_phys = reverse_inp_transform(preds[lead].detach().cpu().float())
            gt_phys = reverse_inp_transform(gt_by_lead[lead].detach().cpu().float())
            metrics = compute_all_metrics(pred_phys, gt_phys, None, variables, lat, lead)
            metrics.pop("bias")
            metrics["dke"] = difference_kinetic_energy(pred_phys, gt_phys, variables,
                                                       lat, str(lead))
            wind = pred_phys[:, wind_idx]
            if ref_winds is not None:
                metrics["dke_pert"] = difference_kinetic_energy(
                    wind, ref_winds[key][lead], wind_vars, lat, str(lead))
            if keep_winds:
                winds[key][lead] = wind
            out[key][lead] = metrics
        print(f"  scored init {key} at leads {sorted(leads)}", flush=True)
    return out, winds


def measure_config(tag, config, groups_map, build_model, loader, leads, variables, lat,
                   reverse_inp_transform, init_key_of, timing_batch,
                   compile_enabled=True, warmup_reps=1, timed_reps=3,
                   ref_winds=None, keep_winds=False, is_linear=DEFAULT_IS_LINEAR):
    model = build_model()
    build_quantised_model(model, groups_map, config, is_linear=is_linear)
    if compile_enabled:
        apply_dynamo_hardening()
        compile_heavy_submodules(model)

    timing_inp, _timing_gt, timing_vars = timing_batch
    timing_inp = timing_inp.to(next(model.parameters()).device)
    timing_steps = max(leads) // INTERVALS[0]

    def rollout_once():
        stormer_rollout_to_leads(model, timing_inp, timing_vars, leads,
                                 intervals=(INTERVALS[0],))

    stats = warmup_and_measure(rollout_once, timing_steps, warmup_reps, timed_reps)
    stats["model_size_gb"] = model_size_gb(model)
    stats["rollout_intervals"] = ",".join(str(i) for i in INTERVALS)
    stats["forwards_per_init"] = forwards_per_init(leads)

    metrics, winds = stormer_metrics_over_inits(
        model, loader, leads, variables, lat, reverse_inp_transform, init_key_of,
        ref_winds=ref_winds, keep_winds=keep_winds)
    return metrics, stats, winds


# --- CHANGE 4 of 4: measured axes, pinned ungated -------------------------------------

CONSISTENCY_AXES = ("balance", "conservation")


def measured_consistency_axes(fp32_run, config_run, tag, lead, dates, noise_floor=None,
                              axes=CONSISTENCY_AXES):
    import ablation_comp as ac
    from distortion import NoiseFloor
    if noise_floor is None:
        noise_floor = NoiseFloor.null()
    specs = ac.available_specs(fp32_run, config_run, ac.metric_registry(fp32_run), lead)
    block = ac.bootstrap_block(len(fp32_run.dates))
    records = ac.sensitivity_records(fp32_run, {tag: config_run}, specs, [tag],
                                     [lead], dates, block, full_table=None,
                                     norm_mode="colmax", noise_floor=noise_floor)
    agg = ac.group_aggregates(records, specs, [tag], lead)[tag]
    return {a: agg[f"{a}_distortion"] for a in axes}


def measured_consistency_axes_all_leads(fp32_run, config_run, tag, leads, dates,
                                        noise_floor=None):
    """One measured_consistency_axes call per requested lead."""
    return {lead: measured_consistency_axes(fp32_run, config_run, tag, lead, dates,
                                            noise_floor)
            for lead in leads}


# --- output ---------------------------------------------------------------------------

RESULTS_CSV_FIELDS = ["tag", "floor", "guide", "predicted_cost", "balance",
                      "conservation", "config", "model_size_gb",
                      "latency_s_per_step", "vram_peak_gb", "vram_reserved_gb",
                      "n_inits", "rollout_intervals", "forwards_per_init"]

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
        w = csv.DictWriter(f, fieldnames=RESULTS_BY_LEAD_CSV_FIELDS,
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in RESULTS_BY_LEAD_CSV_FIELDS})


def write_results_pt(path, per_config_metrics):
    torch.save(per_config_metrics, path)
