"""The dissertation-figure driver."""
# make_figures.py
import physq_path

import argparse
import os

import figure_specs as fs


def with_defaults(argv, defaults):
    supplied = {a.split("=", 1)[0] for a in argv if a.startswith("--")}
    extra = []
    for flag, value in defaults:
        if flag not in supplied:
            extra += [flag, value]
    return argv + extra


def run_registry(specs, roots, outdir):
    written, skipped = [], []
    for s in specs:
        try:
            p = s.build(roots, outdir)
        except Exception as e:
            print(f"  skip {s.id} ({type(e).__name__}: {e})")
            skipped.append(s.id)
            continue
        if p is None:
            skipped.append(s.id)
        else:
            written.append(p)
    return {"written": written, "skipped": skipped}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="section", default=None,
                    choices=("main", "appendix"), help="default: both")
    ap.add_argument("--model", default=None,
                    choices=("aurora", "stormer", "cross"), help="default: all")
    ap.add_argument("--aurora-root", default=".")
    ap.add_argument("--aurora-results", default="harness_results_n48")
    ap.add_argument("--stormer-root", default="../Stormer")
    ap.add_argument("--stormer-results", default="harness_results_n47")
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args(argv)

    roots = {"aurora": fs.Roots(a.aurora_root, a.aurora_results),
             "stormer": fs.Roots(a.stormer_root, a.stormer_results)}
    specs = fs.specs_for(section=a.section, model=a.model)
    print(f"figures -> {a.outdir}/   ({len(specs)} specs)")
    res = run_registry(specs, roots, a.outdir)
    print(f"\n{len(res['written'])} written, {len(res['skipped'])} skipped")
    if res["skipped"]:
        print("  skipped: " + ", ".join(res["skipped"]))
    return res


if __name__ == "__main__":
    main()
