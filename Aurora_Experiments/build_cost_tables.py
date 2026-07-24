import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ablation_scripts"))
from groups import build_groups
from aurora import AuroraPretrained


def main():
    model = AuroraPretrained(autocast=True)          # full-Aurora architecture, CPU
    try:
        # Full pretrained ERA5 model (HF download); only needed for fidelity, not for counts.
        model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
        print("loaded full 0.25 pretrained checkpoint", flush=True)
    except Exception as e:
        print(f"checkpoint load skipped ({type(e).__name__}: {e}); param COUNTS are "
              f"architecture-determined, so cost_tables.pt is unaffected", flush=True)
    groups, params = build_groups(model)   # params = {group: weight_numel}
    table = {g: {"params": int(params[g]), "flops": float(params[g])} for g in groups}
    torch.save(table, "cost_tables.pt")
    for g, v in sorted(table.items()):
        print(f"{g:16} params={v['params']:>12,}")
    print(f"wrote cost_tables.pt ({len(table)} groups)")


if __name__ == "__main__":
    main()
