# merge_harness_results.py
import argparse
import csv
import os
import shutil

import torch

SUFFIX = "__superseded"


def _cfg(row):
    return (row.get("floor", ""), row.get("config", ""))


def plan_merge(old_meta, new_meta):
    """Collisions on a redefined config archive the old tag as `<tag>__superseded`.

    Returns (renames, actions): renames maps a colliding old tag to its archived name,
    actions labels every tag in the output 'new', 'unchanged' or 'old'.
    """
    renames, actions = {}, {}
    for tag in old_meta:
        actions[tag] = "old"
    for tag, meta in new_meta.items():
        if tag in old_meta and old_meta[tag] != meta:
            renames[tag] = tag + SUFFIX
            actions[tag + SUFFIX] = "old"
            actions[tag] = "new"
        elif tag in old_meta:
            actions[tag] = "unchanged"
        else:
            actions[tag] = "new"
    return renames, actions


def merge_pt(old_pt, new_pt, renames, actions):
    out = {}
    for tag, v in old_pt.items():
        out[renames.get(tag, tag)] = v
    for tag, v in new_pt.items():
        if actions.get(tag) in ("new", "unchanged"):
            out[tag] = v
    return out


def merge_csv(old_rows, new_rows, renames, actions):
    """Old rows (superseded ones renamed) plus every new row, replacing by tag.

    Superseded tags are dropped once, up front, so this holds for the one-row files and
    the four-rows-per-tag by-lead files alike.
    """
    replaced = {r["tag"] for r in new_rows if actions.get(r["tag"]) in ("new", "unchanged")}
    out = []
    for r in old_rows:
        r = dict(r)
        r["tag"] = renames.get(r["tag"], r["tag"])
        if r["tag"] in replaced:
            continue
        out.append(r)
    out.extend(dict(r) for r in new_rows
               if actions.get(r["tag"]) in ("new", "unchanged"))
    return out


def _read_csv(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def main(argv=None):
    ap = argparse.ArgumentParser(description="Merge a corrected harness run into an "
                                             "existing results directory.")
    ap.add_argument("--old", default="harness_results")
    ap.add_argument("--new", default="harness_results_corrected")
    ap.add_argument("--out", default="harness_results_merged")
    a = ap.parse_args(argv)

    old_rows = _read_csv(os.path.join(a.old, "harness_results.csv"))
    new_rows = _read_csv(os.path.join(a.new, "harness_results.csv"))
    if not new_rows:
        raise SystemExit(f"no harness_results.csv under {a.new!r} -- nothing to merge")

    old_meta = {r["tag"]: _cfg(r) for r in old_rows}
    new_meta = {r["tag"]: _cfg(r) for r in new_rows}
    renames, actions = plan_merge(old_meta, new_meta)

    os.makedirs(a.out, exist_ok=True)
    old_pt = torch.load(os.path.join(a.old, "harness_results.pt"),
                        map_location="cpu", weights_only=False)
    new_pt = torch.load(os.path.join(a.new, "harness_results.pt"),
                        map_location="cpu", weights_only=False)
    merged = merge_pt(old_pt, new_pt, renames, actions)
    torch.save(merged, os.path.join(a.out, "harness_results.pt"))

    for name in ("harness_results.csv", "harness_results_by_lead.csv",
                 "harness_results_floored.csv", "harness_results_by_lead_floored.csv"):
        o, n = _read_csv(os.path.join(a.old, name)), _read_csv(os.path.join(a.new, name))
        if not o and not n:
            continue
        rows = merge_csv(o, n, renames, actions)
        fields = list(rows[0]) if rows else []
        with open(os.path.join(a.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    print(f"merged {len(old_pt)} old + {len(new_pt)} new -> {len(merged)} configs in {a.out}/")
    for t, arch in sorted(renames.items()):
        print(f"  redefined: {t} -> {arch}")

    for name in ("noise_floor_detailed.pt",):
        src = os.path.join(a.old, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(a.out, name))


if __name__ == "__main__":
    main()
