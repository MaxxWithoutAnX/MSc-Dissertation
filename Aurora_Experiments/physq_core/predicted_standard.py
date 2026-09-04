"""Predicted distortion per tag, for figA09.

Two modes. The default reproduces the frozen RMSE (standard) sidecar from a manifest, under
that manifest's own gate. `--from-results` instead rebuilds all three axes, UNGATED, from the
`config` column of harness_results.csv -- the record of what the harness actually ran.
"""
# predicted_standard.py
import argparse
import csv
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from allocator import load_distortion_table
from frontiers import _axis_sum
from select_harness_configs import decode_config

AXIS = "standard"
LEAD = 120

AURORA_SCHEME_CSVS = {
    "W8A8": {"W8A8": "ablation_analysis/ablations_W8A8/sensitivity.csv"},
    "W8A8_sq": {"W8A8_sq": "ablation_analysis/ablations_W8A8_sq/sensitivity.csv"},
    "W4": {"W4": "ablation_analysis/ablations_W4/sensitivity.csv",
           "W8": "ablation_analysis/ablations_W8/sensitivity.csv"},
}
# 0.01, the practical-equivalence gate regen_all_frontiers.py selected the manifest under.
AURORA_MIN_EFFECT_FRAC = 0.01

STORMER_SCHEME_CSVS = {
    "W8A8": {"W8A8": "ablation_analysis/ablations_W8A8/sensitivity.csv"},
    "W8A8_sq": {"W8A8_sq": "ablation_analysis/ablations_W8A8_sq/sensitivity.csv"},
    "W4": {"W4": "ablation_analysis/ablations_W4/sensitivity.csv",
           "W8": "ablation_analysis/ablations_W8/sensitivity.csv"},
}
# UNGATED: no Stormer noise-floor ensemble exists, so its frontiers were built at 0.0.
STORMER_MIN_EFFECT_FRAC = 0.0

MODELS = {
    "aurora": (AURORA_SCHEME_CSVS, AURORA_MIN_EFFECT_FRAC,
               "harness_configs_corrected.csv", "harness_configs_standard.csv"),
    "stormer": (STORMER_SCHEME_CSVS, STORMER_MIN_EFFECT_FRAC,
                "stormer_harness_configs.csv", "stormer_harness_configs_standard.csv"),
}

UNQUANTISED_FLOOR = "bf16"

FIELDS = ["tag", "lead", AXIS]

# --from-results: {model: (rows read, sidecar written)}. UNGATED on both models -- the gate is
# an allocation policy, not part of the surrogate, and a censored zero is not a prediction.
REBUILT = {
    "aurora": ("harness_results_n48/harness_results.csv",
               "harness_results_n48/figA09_predicted_rebuilt.csv"),
    "stormer": ("harness_results_n47/harness_results.csv",
                "harness_results_n47/figA09_predicted_rebuilt.csv"),
}
REBUILT_AXES = ("balance", "conservation", "standard")
# `floor`, `guide` and `config` are carried so the sidecar can be merged FIRST in
# figure_core._merge_by_tag and still satisfy _floor_reference_tags, which identifies each
# scheme's 100%-damage reference by an empty `config` rather than by a name pattern.
REBUILT_FIELDS = ["tag", "floor", "guide", "lead", "config"] + list(REBUILT_AXES)


def predict_standard(manifest_rows, scheme_csvs_by_floor, lead=LEAD, min_effect_frac=0.0,
                     axes=(AXIS,), carry=False):
    """[{tag, lead, <axes>}] for every row whose floor has a table.

    A row whose floor is absent from `scheme_csvs_by_floor` is OMITTED rather than zeroed: on
    a log-log panel a spurious 0 is dropped by the fit but still reads to the eye as a config
    the surrogate nailed. The bf16 ceiling quantises nothing, so it is 0 by construction.
    `carry` adds the floor/guide/config columns the --from-results sidecar needs."""
    tables = {}
    for floor, csvs in scheme_csvs_by_floor.items():
        tables[floor] = load_distortion_table(csvs, lead=lead, axes=tuple(axes),
                                              min_effect_frac=min_effect_frac)
    out = []
    for row in manifest_rows:
        tag, floor = row.get("tag"), row.get("floor", "")
        if not tag:
            continue
        base = {"tag": tag, "lead": lead}
        if carry:
            base.update(floor=floor, guide=row.get("guide", ""),
                        config=row.get("config") or "")
        if floor == UNQUANTISED_FLOOR:
            out.append({**base, **{a: 0.0 for a in axes}})
            continue
        d = tables.get(floor)
        if d is None:
            continue
        config = decode_config(row.get("config") or "", floor, list(d))
        totals = _axis_sum(d, config, tuple(axes))
        out.append({**base, **{a: float(totals[a]) for a in axes}})
    return out


def write_csv(rows, path, provenance="", fields=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        if provenance:
            f.write("# " + provenance + "\n")
        w = csv.DictWriter(f, fieldnames=fields or FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), default="aurora")
    ap.add_argument("--manifest", default=None, help="override the model's default manifest")
    ap.add_argument("--lead", type=int, default=LEAD)
    ap.add_argument("--out", default=None, help="defaults to the model's sidecar name")
    ap.add_argument("--from-results", action="store_true",
                    help="rebuild all three axes, ungated, from the configs that actually ran")
    a = ap.parse_args(argv)

    scheme_csvs, frac, default_manifest, default_out = MODELS[a.model]
    if a.from_results:
        # The manifest is not consulted at all in this mode: harness_results.csv is the
        # authoritative record of what ran, and on Aurora the two disagree for 14 tags.
        manifest, default_out = REBUILT[a.model]
        frac, axes, fields, carry = 0.0, REBUILT_AXES, REBUILT_FIELDS, True
        prov = ("predicted distortion rebuilt from the RUN configs in "
                f"{os.path.basename(manifest)}  lead={a.lead} min_effect_frac={frac} "
                f"axes={'|'.join(axes)}  n_tags={{n}} of {{m}}")
    else:
        manifest = a.manifest or default_manifest
        axes, fields, carry = (AXIS,), FIELDS, False
        prov = (f"predicted RMSE (standard) axis. model={a.model} manifest={manifest} "
                f"lead={a.lead} min_effect_frac={frac} n_manifest={{m}} n_predicted={{n}}")
    out_path = a.out or default_out
    with open(manifest, newline="") as f:
        rows = [r for r in csv.DictReader(l for l in f if not l.startswith("#"))]

    out = predict_standard(rows, scheme_csvs, lead=a.lead, min_effect_frac=frac,
                           axes=axes, carry=carry)
    write_csv(out, out_path, provenance=prov.format(n=len(out), m=len(rows)), fields=fields)
    print(f"wrote {out_path} ({len(out)} of {len(rows)} rows, "
          f"min_effect_frac={frac})", flush=True)
    skipped = {r["floor"] for r in rows} - set(scheme_csvs) - {UNQUANTISED_FLOOR}
    if skipped:
        print(f"  NO TABLE for floor(s) {sorted(skipped)} -- those tags are omitted",
              flush=True)
    return out


if __name__ == "__main__":
    main()
