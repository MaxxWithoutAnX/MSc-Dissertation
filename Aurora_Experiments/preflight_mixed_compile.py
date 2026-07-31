"""Precision-harness pre-flight: clear both cluster blockers in one short GPU job."""
import argparse
import sys

import torch

import precision_harness as ph
from ablation_scripts.groups import build_groups
from select_harness_configs import decode_config
from run_precision_harness import (build_arg_parser as _harness_parser, build_dataset_accessors,
                                   load_dataset, make_model_builder, select_init_indices)

EAGER_W8A8_VRAM_GB = 31.7          # the eager-fallback signature from the A40 spike
UNIFORM_REF_TAG = "W8A8_floor"     # all-W8A8, the known-good compiled reference
DEFAULT_MIXED_TAG = "W8A8_span1"   # a genuinely mixed int8/bf16 config


def check_truth_schema(ds, leads):
    """(A) Every init in the file must have all `leads` verification steps present."""
    import numpy as np, json
    times = ds.time.values
    idx_of = {t: k for k, t in enumerate(times)}
    init_ts = [np.datetime64(s) for s in json.loads(ds.attrs["init_times"])]
    missing = 0
    for t in init_ts:
        for lead in leads:
            if (t + np.timedelta64(lead, "h")) not in idx_of:
                missing += 1
    ok = missing == 0
    print(f"[A] truth schema: {len(init_ts)} inits x {len(leads)} leads; "
          f"{missing} missing verification step(s) -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def measure_tag(tag, row_by_tag, groups_map, build_model, acc, init0, steps):
    """Build + quantise + compile + warmup + time ONE config (measurement only, no metrics
    pass). Returns (stats, is_mixed, graph_breaks)."""
    from aurora import rollout
    row = row_by_tag[tag]
    floor = row["floor"]
    config = decode_config(row["config"], floor, sorted(groups_map))
    precisions = set(config.values())
    is_mixed = len(precisions) > 1
    model = build_model()
    ph.build_quantised_model(model, groups_map, config)
    ph.apply_dynamo_hardening()
    compiled = ph.compile_heavy_submodules(model)

    def rollout_once():
        batch = acc["make_batch"](init0)
        with torch.inference_mode():
            for _ in rollout(model, batch, steps=steps):
                pass

    from torch._dynamo.utils import counters
    counters.clear()
    stats = ph.warmup_and_measure(rollout_once, steps, warmup_reps=1, timed_reps=2)
    gb = sum(counters.get("graph_break", {}).values())
    print(f"    {tag}: precisions={sorted(precisions)} compiled={compiled} "
          f"graph_breaks={gb}", flush=True)
    print(f"    {tag}: latency={stats['latency_s_per_step']:.2f}s/step  "
          f"peak_vram={stats['vram_peak_gb']:.1f}GB  reserved={stats['vram_reserved_gb']:.1f}GB",
          flush=True)
    del model
    torch.cuda.empty_cache()
    return stats, is_mixed, gb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    hp = _harness_parser()                                   # reuse harness defaults
    ap.add_argument("--manifest", default=hp.get_default("manifest"))
    ap.add_argument("--data", default=hp.get_default("data"))
    ap.add_argument("--checkpoint", default=hp.get_default("checkpoint"))
    ap.add_argument("--mixed-tag", default=DEFAULT_MIXED_TAG, dest="mixed_tag")
    ap.add_argument("--steps", type=int, default=8)          # reaches lead 48h; enough to peak
    ap.add_argument("--leads", default="24,72,120,168")
    a = ap.parse_args(argv)
    leads = [int(x) for x in a.leads.split(",")]

    if not torch.cuda.is_available():
        print("FAIL: no CUDA -- run this on an a40 node", flush=True)
        return 2

    ds = load_dataset(a.data, synthetic_ok=False)
    schema_ok = check_truth_schema(ds, leads)

    row_by_tag = {r["tag"]: r for r in ph.read_manifest(a.manifest)}
    for t in (UNIFORM_REF_TAG, a.mixed_tag):
        if t not in row_by_tag:
            print(f"FAIL: {t} not in {a.manifest}", flush=True)
            return 2

    build_model = make_model_builder(a.checkpoint, "cuda")
    template = build_model()
    groups_map, _ = build_groups(template)
    del template
    acc = build_dataset_accessors(ds)
    init0 = select_init_indices(ds, 0)[0]

    print("[B] mixed int8/bf16 compile A/B:", flush=True)
    ref, _, ref_gb = measure_tag(UNIFORM_REF_TAG, row_by_tag, groups_map, build_model, acc,
                                 init0, a.steps)
    mix, is_mixed, mix_gb = measure_tag(a.mixed_tag, row_by_tag, groups_map, build_model, acc,
                                        init0, a.steps)

    if not is_mixed:
        print(f"FAIL: {a.mixed_tag} is not actually mixed precision", flush=True)
        return 2

    vram_ok = mix["vram_peak_gb"] <= 1.3 * ref["vram_peak_gb"] and mix["vram_peak_gb"] < 26.0
    lat_ok = mix["latency_s_per_step"] <= 1.5 * ref["latency_s_per_step"]
    not_eager = mix["vram_peak_gb"] < 0.85 * EAGER_W8A8_VRAM_GB
    compile_ok = vram_ok and lat_ok and not_eager

    print("\n=== PRE-FLIGHT SUMMARY ===", flush=True)
    print(f"  [A] 2021 truth schema           : {'PASS' if schema_ok else 'FAIL'}", flush=True)
    print(f"  [B] mixed vs uniform VRAM        : {mix['vram_peak_gb']:.1f} vs "
          f"{ref['vram_peak_gb']:.1f} GB ({'PASS' if vram_ok else 'FAIL'})", flush=True)
    print(f"  [B] mixed vs uniform latency     : {mix['latency_s_per_step']:.2f} vs "
          f"{ref['latency_s_per_step']:.2f} s/step ({'PASS' if lat_ok else 'FAIL'})", flush=True)
    print(f"  [B] not eager-fallback (<{0.85*EAGER_W8A8_VRAM_GB:.0f}GB): "
          f"{'PASS' if not_eager else 'FAIL'}", flush=True)
    ok = schema_ok and compile_ok
    print(f"\n  OVERALL: {'PASS -- clear to launch the full sweep' if ok else 'FAIL -- investigate'}",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
