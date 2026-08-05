import csv
import os


def encode_config(config, floor):
    """{group: precision} -> 'prec:g|g;prec:g' listing only NON-floor groups (floor is
    implicit). Precisions and groups sorted for determinism; empty string = all-floor."""
    by_prec = {}
    for g, p in config.items():
        if p != floor:
            by_prec.setdefault(p, []).append(g)
    return ";".join(f"{p}:" + "|".join(sorted(by_prec[p])) for p in sorted(by_prec, key=str.lower))


def decode_config(s, floor, all_groups):
    """Inverse of encode_config: every group starts at floor; non-floor groups overridden."""
    config = {g: floor for g in all_groups}
    for part in filter(None, s.split(";")):
        prec, groups = part.split(":")
        for g in groups.split("|"):
            config[g] = prec
    return config


def _mentioned_groups(row):
    if row.get("config"):
        return {g for part in row["config"].split(";") if part
                for g in part.split(":")[1].split("|")}
    if row.get("protected"):
        return set(row["protected"].split("|"))
    return set()


def _num(row, key):
    v = row.get(key)
    return float(v) if v not in (None, "") else None


def load_frontier(csv_path, floor):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    all_groups = sorted(set().union(*(_mentioned_groups(r) for r in rows)) if rows else set())
    out = []
    for r in rows:
        if r.get("config") is not None:
            cfg = decode_config(r["config"], floor, all_groups)
        else:
            prot = set(r["protected"].split("|")) if r.get("protected") else set()
            cfg = {g: ("bf16" if g in prot else floor) for g in all_groups}
        out.append({"cost": _num(r, "cost"), "balance": _num(r, "balance"),
                    "conservation": _num(r, "conservation"), "config": cfg, "floor": floor})
    return out


KNEE_TOL = 0.05


def _knee_index(rows_sorted, tol=KNEE_TOL):
    if len(rows_sorted) < 3:
        return len(rows_sorted) // 2
    drops = [_objective(rows_sorted[i - 1], "balance") - _objective(rows_sorted[i], "balance")
             for i in range(1, len(rows_sorted))]
    best = max(drops)
    if best <= 0:
        return len(rows_sorted) // 2
    tied = [i for i, d in enumerate(drops) if d >= (1.0 - tol) * best]
    return min(tied, key=lambda i: _cost(rows_sorted[i + 1]) - _cost(rows_sorted[i])) + 1


def _cost(r):
    return r["cost"] if r["cost"] is not None else 0.0


def _objective(r, key, default=0.0):
    v = r.get(key, default)
    if v is None or v == "":
        return default
    return float(v)


def _useful_cost_max(rs, objectives=("balance", "conservation")):
    present = [o for o in objectives if any(o in r for r in rs)]
    if not present:
        return _cost(rs[-1])
    mins = {o: min(_objective(r, o) for r in rs if o in r) for o in present}
    for r in rs:                                   # rs is cost-sorted
        if all(o not in r or _objective(r, o) <= mins[o] for o in present):
            return _cost(r)
    return _cost(rs[-1])


def pick_anchors_and_span(rows, k=3, cost_tol=0.02, span_over_useful_range=True):
    rs = sorted(rows, key=_cost)
    labelled = [("floor", rs[0]), ("ceiling", rs[-1]), ("knee", rs[_knee_index(rs)])]
    cmin = _cost(rs[0])
    cmax = _useful_cost_max(rs) if span_over_useful_range else _cost(rs[-1])
    for i in range(1, k + 1):
        target = cmin + (cmax - cmin) * i / (k + 1)
        band = [r for r in rs
                if abs(_cost(r) - target) <= cost_tol * max(abs(target), 1e-12)]
        if not band:                           # tolerance too tight: use nearest cost
            band = [min(rs, key=lambda r: abs(_cost(r) - target))]
        labelled.append((f"span{i}", min(
            band, key=lambda r: (_objective(r, "balance"), _cost(r),
                                 sorted(r["config"].items())))))
    seen, out = set(), []
    for label, r in labelled:
        key = frozenset(r["config"].items())
        if key not in seen:
            seen.add(key)
            out.append(dict(r, tag=label))
    return out


def dedup(configs):
    """Global dedup by (floor, frozenset(config)); first occurrence wins. An all-bf16 config
    is floor-independent (same model however you build it), so its floor is canonicalised to
    'bf16' before keying -- this collapses the shared full-precision ceiling across schemes."""
    seen, out = set(), []
    for c in configs:
        floor = "bf16" if set(c["config"].values()) == {"bf16"} else c["floor"]
        key = (floor, frozenset(c["config"].items()))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def match_rmse(phys_selected, rmse_rows):
    """For each selected physics config, the RMSE row at nearest cost (deduped)."""
    seen, out = set(), []
    for p in phys_selected:
        r = min(rmse_rows, key=lambda rr: abs(_cost(rr) - _cost(p)))
        key = frozenset(r["config"].items())
        if key not in seen:
            seen.add(key)
            out.append(dict(r, tag="rmse_" + p["tag"]))
    return out


def match_at_budget(phys_selected, order_rows, tag_prefix="rmse_", direction="down"):
    if direction not in ("down", "up"):
        raise ValueError(f"direction must be 'down' or 'up', got {direction!r}")
    out = []
    for p in phys_selected:
        if direction == "down":
            cands = [r for r in order_rows if _cost(r) <= _cost(p) + 1e-12]
            r = max(cands, key=_cost) if cands else min(order_rows, key=_cost)
        else:
            cands = [r for r in order_rows if _cost(r) >= _cost(p) - 1e-12]
            r = min(cands, key=_cost) if cands else max(order_rows, key=_cost)
        out.append(dict(r, tag=tag_prefix + p["tag"], floor=p["floor"]))
    return out


def uniform_endpoints(rows, floor):
    """floor (all-floor, cost 0) as floor_<floor>; ceiling (all-bf16) once, floor='bf16'."""
    rs = sorted(rows, key=_cost)
    floor_row = dict(rs[0], tag="floor", floor=floor)
    ceiling = {"cost": rs[-1]["cost"], "balance": rs[-1]["balance"],
               "conservation": rs[-1]["conservation"],
               "config": {g: "bf16" for g in rs[-1]["config"]},
               "floor": "bf16", "tag": "ceiling"}
    return [floor_row, ceiling]


def random_configs(random_rows, near_cost, k=4):
    """The k random rows nearest near_cost (deduped by config)."""
    seen, out = set(), []
    for r in sorted(random_rows, key=lambda rr: abs(_cost(rr) - near_cost)):
        key = frozenset(r["config"].items())
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(r, tag=f"rand_{len(out)}"))
        if len(out) == k:
            break
    return out


def probe_configs(all_groups, floor, divergent=("dec2_attn", "upsample"),
                  rmse_favoured=("encoder_io", "film")):
    """Off-frontier localisation probes: protect ONLY the named groups to bf16. Unknown
    group name raises (guards typos)."""
    for g in (*divergent, *rmse_favoured):
        if g not in all_groups:
            raise ValueError(f"probe group {g!r} not in scheme groups")

    def mk(groups, tag):
        cfg = {g: ("bf16" if g in groups else floor) for g in all_groups}
        return {"cost": None, "balance": None, "conservation": None,
                "config": cfg, "floor": floor, "tag": tag}
    return [mk(set(divergent), "probe_divergent"),
            mk(set(rmse_favoured), "probe_rmse_favoured")]


SCHEMES = {
    "W8A8": dict(floor="W8A8",
                 physics="frontier_flops_real/frontier_B_W8A8_physics.csv",
                 rmse="frontier_flops_real/frontier_B_W8A8_rmse.csv",
                 random="frontier_flops_real/random_W8A8.csv"),
    "W8A8_sq": dict(floor="W8A8_sq",
                    physics="frontier_flops_real/frontier_B_W8A8_sq_physics.csv",
                    rmse="frontier_flops_real/frontier_B_W8A8_sq_rmse.csv"),
    "W4W8": dict(floor="W4",
                 physics="frontier_flops_real/frontier_A_W4W8_physics.csv",
                 rmse="frontier_flops_real/frontier_A_W4W8_rmse.csv"),
}

SCHEMES_CORRECTED = {
    "W8A8": dict(floor="W8A8",
                 physics="frontier_corrected/frontier_B_W8A8_physics.csv",
                 rmse="frontier_corrected/frontier_B_W8A8_rmse.csv",
                 order="frontier_corrected/guide_order_B_W8A8_rmse.csv",
                 random="frontier_corrected/random_W8A8.csv"),
    "W8A8_sq": dict(floor="W8A8_sq",
                    physics="frontier_corrected/frontier_B_W8A8_sq_physics.csv",
                    rmse="frontier_corrected/frontier_B_W8A8_sq_rmse.csv",
                    order="frontier_corrected/guide_order_B_W8A8_sq_rmse.csv"),
    "W4W8": dict(floor="W4",
                 physics="frontier_corrected/frontier_A_W4W8_physics.csv",
                 rmse="frontier_corrected/frontier_A_W4W8_rmse.csv",
                 order="frontier_corrected/guide_order_A_W4W8_rmse.csv"),
}


def select_scheme(scheme, spec, k=3, with_extras=False):
    """All configs for one scheme: physics anchors+span, cost-matched RMSE, uniform
    endpoints, and (if with_extras) random + localisation probes. Tags scheme-prefixed
    (except the shared `ceiling`)."""
    floor = spec["floor"]
    phys = load_frontier(spec["physics"], floor)
    rmse = load_frontier(spec["rmse"], floor)
    picks = pick_anchors_and_span(phys, k)
    for p in picks:
        p["guide"] = "physics"
    interior = [p for p in picks if p["tag"] not in ("floor", "ceiling")]  # endpoints come from uniform_endpoints
    if spec.get("order"):
        order_rows = load_frontier(spec["order"], floor)
        rmatch = match_at_budget(interior, order_rows, direction="down")
        rmatch += match_at_budget(interior, order_rows, tag_prefix="rmseup_",
                                  direction="up")
    else:
        rmatch = match_rmse(interior, rmse)
    for r in rmatch:
        r["guide"] = "rmse"
    unif = uniform_endpoints(phys, floor)
    for u in unif:
        u["guide"] = "uniform"
    configs = unif + picks + rmatch
    if with_extras:
        all_groups = sorted(phys[-1]["config"])          # ceiling row lists all groups
        probes = probe_configs(all_groups, floor)
        for pr in probes:
            pr["guide"] = "probe"
        rnd = []
        if spec.get("random"):
            interior = [p for p in picks if p["tag"] not in ("floor", "ceiling")]
            costs = sorted(_cost(p) for p in interior) or [0.0]
            near = costs[len(costs) // 2]           # mid-frontier cost, not the ~free knee
            rnd = random_configs(load_frontier(spec["random"], floor), near)
            for x in rnd:
                x["guide"] = "random"
        configs += probes + rnd
    for c in configs:
        if c["tag"] != "ceiling":
            c["tag"] = f"{scheme}_{c['tag']}"
    return configs


def select_all(k=3, extras_scheme="W8A8", outdir=".", schemes=None,
               manifest_name="harness_configs.csv"):
    schemes = schemes or SCHEMES
    configs = []
    for scheme, spec in schemes.items():
        configs += select_scheme(scheme, spec, k=k, with_extras=(scheme == extras_scheme))
    kept = dedup(configs)
    kept_tags = {c["tag"] for c in kept}
    dropped = [c["tag"] for c in configs if c["tag"] not in kept_tags]
    if dropped:
        print(f"  dedup collapsed {len(dropped)} duplicate config(s): {', '.join(dropped)}")
    os.makedirs(outdir, exist_ok=True)
    write_manifest(os.path.join(outdir, manifest_name), kept)
    return kept


def write_manifest(path, configs):
    def fmt(v):
        return "" if v is None else f"{v:.6g}"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "floor", "guide", "cost", "balance", "conservation", "config"])
        for c in configs:
            w.writerow([c["tag"], c["floor"], c["guide"], fmt(c["cost"]),
                        fmt(c["balance"]), fmt(c["conservation"]),
                        encode_config(c["config"], c["floor"])])


def main():
    cfgs = select_all()
    print(f"wrote harness_configs.csv ({len(cfgs)} unique models)")
    by = {}
    for c in cfgs:
        by[c["guide"]] = by.get(c["guide"], 0) + 1
    for g, n in sorted(by.items()):
        print(f"  {g:8} {n}")


if __name__ == "__main__":
    main()
