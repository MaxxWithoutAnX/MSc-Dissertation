"""Read back the headline numbers from two results dirs."""
import physq_path

import argparse
import csv
import os

MISSING = "--"


def _rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [r for r in csv.DictReader(line for line in f if not line.startswith("#"))]


def _find(rows, **eq):
    """First row matching every key=value (string-compared, so 120 and '120' both work)."""
    for r in rows:
        if all(str(r.get(k, "")).strip() == str(v) for k, v in eq.items()):
            return r
    return None


def _f(row, key, fmt="{:.3g}"):
    if row is None or key not in row:
        return MISSING
    try:
        return fmt.format(float(row[key]))
    except (TypeError, ValueError):
        return str(row[key]) or MISSING


def _ci(row, lo="ci_lo", hi="ci_hi"):
    if row is None:
        return MISSING
    return f"[{_f(row, lo)}, {_f(row, hi)}]"


def collect(d):
    """{label: value-string} for every caption-bearing quantity found in results dir `d`."""
    out = {}
    rem = _rows(os.path.join(d, "harness_damage_removed.csv"))
    remc = _rows(os.path.join(d, "harness_damage_removed_dry_air_mass.csv"))
    for tag in ("W8A8_span1", "W8A8_knee", "W8A8_rmse_knee"):
        r = _find(rem, tag=tag, block="2")
        out[f"damage removed, balance      {tag}"] = f"{_f(r, 'removed_pct')}% {_ci(r)}"
        rc = _find(remc, tag=tag, block="2")
        out[f"damage removed, conservation {tag}"] = f"{_f(rc, 'removed_pct')}% {_ci(rc)}"

    comp = _rows(os.path.join(d, "harness_results_composite.csv"))
    for pair in ("W8A8_knee vs W8A8_rmse_knee", "W8A8_span1 vs W8A8_rmse_span1",
                 "W4W8_span1 vs W4W8_rmse_span1"):
        r = _find(comp, pair=pair) or _find(comp, pair=pair.replace(" vs ", "|"))
        out[f"composite ratio  {pair}"] = f"{_f(r, 'ratio')} {_ci(r)}"

    anti = _rows(os.path.join(d, "harness_results_anticontrol_composite.csv"))
    for r in anti:
        key = r.get("pair") or r.get("tag") or "?"
        out[f"anticontrol (composite)  {key}"] = f"{_f(r, 'ratio')} {_ci(r)}"

    var = _rows(os.path.join(d, "harness_axis_variants.csv"))
    by = {}
    for r in var:
        by.setdefault(r.get("variant", "?"), {})[r.get("tag", "?")] = r.get("value")
    for variant in ("dry_only", "dry_and_negq", "multi_family"):
        v = by.get(variant, {})
        try:
            ratio = float(v["W8A8_rmse_span1"]) / float(v["W8A8_span1"])
            out[f"axis variant span1 advantage  {variant}"] = f"{ratio:.3g}x"
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            out[f"axis variant span1 advantage  {variant}"] = MISSING

    lead = _rows(os.path.join(d, "aurora_lead_robustness.csv"))
    for r in lead:
        key = r.get("quantity") or r.get("pair") or r.get("lead") or "?"
        out[f"lead robustness  {key}"] = f"{_f(r, 'value') if 'value' in r else _f(r, 'ratio')} {_ci(r)}"

    byl = _rows(os.path.join(d, "harness_results_by_lead.csv"))
    for L in (24, 72, 120, 168):
        f_ = _find(byl, tag="W8A8_floor", lead=L)
        s_ = _find(byl, tag="W8A8_span1", lead=L)
        try:
            out[f"balance ratio floor/span1 @{L}h"] = \
                f"{float(f_['balance']) / float(s_['balance']):.4g}x"
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            out[f"balance ratio floor/span1 @{L}h"] = MISSING
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", default="harness_results_merged")
    ap.add_argument("--new", default="harness_results_n48")
    a = ap.parse_args(argv)

    old, new = collect(a.old), collect(a.new)
    keys = list(dict.fromkeys(list(old) + list(new)))
    w = max(len(k) for k in keys) + 2
    print(f"{'quantity':{w}}{'OLD (' + a.old + ')':<34}{'NEW (' + a.new + ')':<34}  moved")
    print("-" * (w + 74))
    for k in keys:
        o, n = old.get(k, MISSING), new.get(k, MISSING)
        flag = "" if o == n else ("  <-- CHECK CAPTION" if MISSING not in (o, n) else "  (partial)")
        print(f"{k:{w}}{o:<34}{n:<34}{flag}")
    return {"old": old, "new": new}


if __name__ == "__main__":
    main()
