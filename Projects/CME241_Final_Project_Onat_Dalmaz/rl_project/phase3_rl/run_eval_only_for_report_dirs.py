#!/usr/bin/env python3
"""
Eval-only pass for existing Phase 3 report dirs: load Q, rebuild DP_PHASE2 from train data,
re-run all baselines + DP_PHASE2, recompute Delta_fair from {DP_PHASE2, A, B, Hold}, rewrite EVAL_SUMMARY.
Usage: python -m phase3_rl.run_eval_only_for_report_dirs --outdirs /path/to/cfg6 /path/to/cfg2
"""

import json
import subprocess
import sys
from pathlib import Path


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Eval-only pass for Phase 3 report dirs")
    ap.add_argument("--outdirs", type=str, nargs="+", required=True, help="Full paths to report outdirs")
    ap.add_argument("--dp_phase2_reward_mode", type=str, default="zero_mean", choices=["drift", "zero_mean"], help="DP_PHASE2 reward: zero_mean (fair) or drift")
    args = ap.parse_args()

    for outdir in [Path(p) for p in args.outdirs]:
        manifest_path = outdir / "RUN_MANIFEST.json"
        if not manifest_path.exists():
            print("Skip (no RUN_MANIFEST):", outdir)
            continue
        manifest = json.loads(manifest_path.read_text())
        run_args = manifest.get("args", {})
        run_root = run_args.get("run_root")
        if not run_root:
            print("Skip (no run_root in manifest):", outdir)
            continue
        # Build CLI args from manifest; override resume_eval=1 and dp_phase2_reward_mode
        cmd = [sys.executable, "-m", "phase3_rl.cli_phase3", "--run_root", run_root, "--outdir", str(outdir)]
        for k, v in run_args.items():
            if k in ("run_root", "outdir"):
                continue
            if k == "resume_eval":
                continue
            if k == "dp_phase2_reward_mode":
                continue
            if isinstance(v, bool):
                if v:
                    cmd.append(f"--{k}")
            else:
                cmd.append(f"--{k}")
                cmd.append(str(v))
        cmd.extend(["--resume_eval", "1", "--dp_phase2_reward_mode", getattr(args, "dp_phase2_reward_mode", "zero_mean")])
        print("Running eval-only for", outdir.name, "dp_phase2_reward_mode=%s ..." % getattr(args, "dp_phase2_reward_mode", "zero_mean"))
        r = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent, timeout=360000)
        if r.returncode != 0:
            print("Failed:", outdir.name, file=sys.stderr)
        else:
            print("OK:", outdir.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
