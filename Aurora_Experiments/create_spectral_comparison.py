import numpy as np
import matplotlib.pyplot as plt

from plot_common import (load_run, RUNS, RUN_COLORS, SPECTRAL_VARS,
                         metric_registry, per_init_series, ensure_dir)

OUT = ensure_dir("plots_spectral_comparison")
PAIRS = [("SpecRes", "SpecResLog"), ("SpecDiv>", "SpecDivW1")]

# Left/right y-axis labels for each (existing, paper) metric pair.
AXIS_LABELS = {
    ("SpecRes", "SpecResLog"): ("SpecRes (normalized-linear)", "SpecResLog (log-raw)"),
    ("SpecDiv>", "SpecDivW1"): ("SpecDiv> (KL)", "SpecDivW1 (Wasserstein)"),
}

LEFT_AXIS_COLOR = "black"
RIGHT_AXIS_COLOR = "dimgray"


def _series_by_label(run):
    """{label: MetricSpec} for this run's registry, for direct lookup."""
    return {s.label: s for s in metric_registry(run)}


def main():
    runs = {}
    for key in RUNS:
        try:
            runs[key] = load_run(key)
        except FileNotFoundError:
            print(f"skip {key}: no all_metrics_{key}.pt")
    if not runs:
        raise SystemExit("no runs found")

    any_key, any_run = next(iter(runs.items()))
    leads = any_run.leads

    for key, run in list(runs.items()):
        if run.leads != leads:
            print(f"skip {key}: leads {run.leads} differ from {any_key}'s {leads}, "
                  f"cannot share the comparison's x-axis")
            del runs[key]
    if not runs:
        raise SystemExit("no runs left with leads matching the reference run")

    varlabels = list(SPECTRAL_VARS)
    # Hoisted once per run (was previously rebuilt inside the innermost loop).
    run_specs = {key: _series_by_label(run) for key, run in runs.items()}

    nrow = len(varlabels)
    ncol = len(PAIRS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 3.2 * nrow),
                             squeeze=False)

    for r, vlabel in enumerate(varlabels):
        for c, (base, paper) in enumerate(PAIRS):
            ax = axes[r][c]
            ax2 = ax.twinx()
            base_axis_label, paper_axis_label = AXIS_LABELS[(base, paper)]

            for key, run in runs.items():
                specs = run_specs[key]
                color = RUN_COLORS.get(key, "gray")

                base_name = f"{base} {vlabel}"
                if base_name not in specs:
                    print(f"warning: run {key} has no '{base_name}' in its metric "
                          f"registry -- skipping (absent, not a ~0 value)")
                else:
                    means = np.array([per_init_series(run, specs[base_name], lt).mean()
                                      for lt in leads])
                    ax.plot(leads, means, ":", color=color,
                            label=f"{key} {base} (left)")

                paper_name = f"{paper} {vlabel}"
                if paper_name not in specs:
                    print(f"warning: run {key} has no '{paper_name}' in its metric "
                          f"registry -- skipping (absent, not a ~0 value)")
                else:
                    means = np.array([per_init_series(run, specs[paper_name], lt).mean()
                                      for lt in leads])
                    ax2.plot(leads, means, "-", color=color,
                             label=f"{key} {paper} (right)")

            ax.set_title(f"{vlabel}: {base} (dotted, left axis) vs {paper} (solid, right axis)")
            ax.set_xlabel("lead time (h)")
            ax.set_ylabel(base_axis_label, color=LEFT_AXIS_COLOR)
            ax.tick_params(axis="y", labelcolor=LEFT_AXIS_COLOR)
            ax2.set_ylabel(paper_axis_label, color=RIGHT_AXIS_COLOR, style="italic")
            ax2.tick_params(axis="y", labelcolor=RIGHT_AXIS_COLOR)

            if r == 0 and c == 0:
                legend_handles = ax.get_lines() + ax2.get_lines()
                legend_labels = [h.get_label() for h in legend_handles]

    fig.tight_layout()
    fig.legend(legend_handles, legend_labels, fontsize=7, ncol=len(legend_labels),
               loc="lower center", bbox_to_anchor=(0.5, 1.0))
    path = f"{OUT}/spectral_metric_comparison.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
