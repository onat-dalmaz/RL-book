#!/usr/bin/env python3
"""
Create final Phase 3 SNX bundle: phase3_sweep_snx/, phase3_report_real_snx_cfg*/, FINAL_SUMMARY.md, MANIFEST_FINAL.json.
Excludes any report dir with INVALID_RUN.txt or missing EFFECTIVE_CONFIG.json.
"""

import argparse
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from datetime import datetime, timezone


def parse_args():
    ap = argparse.ArgumentParser(description="Bundle Phase 3 SNX results for submission")
    ap.add_argument("--results_dir", type=str, default="/home/ubuntu/onat/results", help="Results root")
    ap.add_argument("--bundles_dir", type=str, default="/home/ubuntu/bundles", help="Output dir for tarball")
    ap.add_argument("--sweep_name", type=str, default="phase3_sweep_snx")
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--extra_include", type=str, nargs="*", default=[], help="Extra report dir names to include (e.g. phase3_report_real_snx_cfg2_HEADLINE_v1)")
    ap.add_argument("--bundle_suffix", type=str, default="", help="Optional suffix for tarball name, e.g. FIXDP -> phase3_snx_results_FINAL_HEADLINE_FIXDP_<ts>.tar.gz")
    return ap.parse_args()


def git_hash():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    bundles_dir = Path(args.bundles_dir)
    bundles_dir.mkdir(parents=True, exist_ok=True)

    sweep_dir = results_dir / args.sweep_name
    if not sweep_dir.exists() or not (sweep_dir / "TOP_CONFIGS.json").exists():
        print("Missing sweep or TOP_CONFIGS.json", sweep_dir, file=sys.stderr)
        return 1

    top_configs = json.loads((sweep_dir / "TOP_CONFIGS.json").read_text())[: args.top_k]
    report_dirs = []
    seen = set()
    for rec in top_configs:
        cid, cm, p0, p1 = rec["config_id"], rec["c_maker"], rec["p0"], rec["p1"]
        d = results_dir / f"phase3_report_real_snx_cfg{cid}_cm{cm}_p0{p0}_p1{p1}"
        if d.exists() and str(d) not in seen:
            if (d / "INVALID_RUN.txt").exists():
                print("Excluding invalid:", d)
                continue
            if not (d / "EFFECTIVE_CONFIG.json").exists():
                print("Excluding (no EFFECTIVE_CONFIG):", d)
                continue
            report_dirs.append(d)
            seen.add(str(d))
    for name in getattr(args, "extra_include", []) or []:
        d = results_dir / name
        if d.exists() and str(d) not in seen:
            if (d / "INVALID_RUN.txt").exists():
                print("Excluding invalid (extra):", d)
                continue
            if not (d / "EFFECTIVE_CONFIG.json").exists():
                print("Excluding (no EFFECTIVE_CONFIG, extra):", d)
                continue
            report_dirs.append(d)
            seen.add(str(d))

    # Build FINAL_SUMMARY.md
    summary_lines = [
        "# Phase 3 SNX — Final summary",
        "",
        "## Sweep",
        f"- Dir: {sweep_dir.name}/",
        f"- Top configs: {[r['config_id'] for r in top_configs]}",
        "",
    ]
    for d in report_dirs:
        name = d.name
        summary_lines.append(f"## {name}")
        if (d / "EVAL_SUMMARY.csv").exists():
            import pandas as pd
            import math
            def _v(x, default="—"):
                if x is None or x == "" or (isinstance(x, float) and math.isnan(x)):
                    return default
                return x
            df = pd.read_csv(d / "EVAL_SUMMARY.csv")
            ql = df[df["policy"] == "QL"]
            dp_exact = df[df["policy"] == "DP_EXACT"]
            if dp_exact.empty:
                dp_exact = df[df["policy"] == "DP"]
            dp_emp = df[df["policy"] == "DP_EMPIRICAL"]
            dp_phase2 = df[df["policy"] == "DP_PHASE2"]
            if not ql.empty:
                q = ql.iloc[0]
                summary_lines.append(f"- QL mean: {_v(q.get('mean_cum_bps'))} CI [{_v(q.get('ci_low'))}, {_v(q.get('ci_high'))}]")
            if not dp_exact.empty:
                dpr = dp_exact.iloc[0]
                summary_lines.append(f"- DP_exact (upper bound) mean: {_v(dpr.get('mean_cum_bps'))} CI [{_v(dpr.get('ci_low'))}, {_v(dpr.get('ci_high'))}]")
            if not dp_phase2.empty:
                dpr = dp_phase2.iloc[0]
                summary_lines.append(f"- DP_PHASE2 (fair baseline) mean: {_v(dpr.get('mean_cum_bps'))} CI [{_v(dpr.get('ci_low'))}, {_v(dpr.get('ci_high'))}]")
            if not dp_emp.empty:
                dpr = dp_emp.iloc[0]
                summary_lines.append(f"- DP_empirical mean: {_v(dpr.get('mean_cum_bps'))} CI [{_v(dpr.get('ci_low'))}, {_v(dpr.get('ci_high'))}]")
            if not ql.empty:
                q = ql.iloc[0]
                summary_lines.append(f"- **Delta_fair** (QL − best of DP_PHASE2, A, B, Hold): {_v(q.get('delta_fair_mean'))} CI [{_v(q.get('delta_fair_ci_low'))}, {_v(q.get('delta_fair_ci_high'))}]")
                summary_lines.append(f"- **Gap_to_oracle** (QL − DP_exact): {_v(q.get('gap_to_oracle_mean'))} CI [{_v(q.get('gap_to_oracle_ci_low'))}, {_v(q.get('gap_to_oracle_ci_high'))}]")
        summary_lines.append("")

    summary_path = results_dir / "phase3_snx_results_FINAL_SUMMARY.md"
    summary_path.write_text("\n".join(summary_lines))

    analysis_path = results_dir / "phase3_snx_RESULTS_ANALYSIS_AND_INTERPRETATION.md"
    if not analysis_path.exists():
        analysis_path = None

    manifest = {
        "git_hash": git_hash(),
        "created": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "included_dirs": [str(sweep_dir.relative_to(results_dir))] + [str(d.relative_to(results_dir)) for d in report_dirs],
        "top_configs": top_configs,
    }

    manifest_path = results_dir / "MANIFEST_FINAL.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Tarball
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    has_extra = getattr(args, "extra_include", None) and len(args.extra_include) > 0
    suffix = getattr(args, "bundle_suffix", "") or ""
    if suffix:
        tarball = bundles_dir / f"phase3_snx_results_FINAL_HEADLINE_{suffix}_{ts}.tar.gz"
    elif has_extra:
        tarball = bundles_dir / f"phase3_snx_results_FINAL_HEADLINE_{ts}.tar.gz"
    else:
        tarball = bundles_dir / f"phase3_snx_results_FINAL_{ts}.tar.gz"
    with tarfile.open(tarball, "w:gz") as tf:
        tf.add(sweep_dir, arcname=sweep_dir.name)
        for d in report_dirs:
            tf.add(d, arcname=d.name)
        tf.add(summary_path, arcname=summary_path.name)
        tf.add(manifest_path, arcname=manifest_path.name)
        if analysis_path is not None and analysis_path.exists():
            tf.add(analysis_path, arcname=analysis_path.name)
    print("Created", tarball)
    return 0


if __name__ == "__main__":
    sys.exit(main())
