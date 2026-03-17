#!/usr/bin/env python3
"""
Phase 3 SNX Run Harvest: discover runs, extract metrics, dedupe, produce RUN_REGISTRY,
tables, run cards, and optional figures.
Usage:
  python3 -m phase3_rl.harvest_results \\
    --results_root /home/ubuntu/onat/results \\
    --bundles_root /home/ubuntu/bundles \\
    --outdir /home/ubuntu/onat/results/REPORT_HARVEST \\
    --include_figures 1
"""
import argparse
import sys
from pathlib import Path

from . import _harvest_lib as H


def main():
    ap = argparse.ArgumentParser(description="Phase 3 harvest: index runs, tables, run cards")
    ap.add_argument("--results_root", type=str, default="/home/ubuntu/onat/results")
    ap.add_argument("--bundles_root", type=str, default="/home/ubuntu/bundles")
    ap.add_argument("--outdir", type=str, default="/home/ubuntu/onat/results/REPORT_HARVEST")
    ap.add_argument("--include_figures", type=int, default=1)
    ap.add_argument("--skip_extract", type=int, default=0, help="1 = skip bundle extraction")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    bundles_root = Path(args.bundles_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Step 1 — Extract bundles
    bundles_extracted = outdir / "bundles_extracted"
    if not args.skip_extract and bundles_root.exists():
        extracted = H.extract_all_bundles(bundles_root, bundles_extracted)
        print("Extracted %d bundles to %s" % (len(extracted), bundles_extracted))
    else:
        bundles_extracted = outdir / "bundles_extracted" if (outdir / "bundles_extracted").exists() else None

    # Step 2 — Discover run dirs
    run_dirs = H.discover_run_dirs(results_root, bundles_extracted)
    # De-dupe by resolved path
    seen = set()
    unique = []
    for path, source in run_dirs:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append((path, source))
    run_dirs = unique
    print("Discovered %d run directories" % len(run_dirs))

    # Step 3 — Parse into registry
    registry = []
    for path, source in run_dirs:
        try:
            r = H.normalize_run(path, source)
            if r.get("invalid_run"):
                r["notes"] = (r.get("notes") or "") + "; INVALID_RUN.txt"
            registry.append(r)
        except Exception as e:
            print("Skip %s: %s" % (path, e), file=sys.stderr)

    # Sweep dirs (top-level only) for TABLE_SWEEP_SUMMARY
    sweep_dirs = [results_root / "phase3_sweep_snx"]
    if bundles_extracted and bundles_extracted.exists():
        for b in bundles_extracted.iterdir():
            if b.is_dir():
                for c in b.iterdir():
                    if c.is_dir() and c.name == "phase3_sweep_snx":
                        sweep_dirs.append(c)

    # Step 4 — Canonical selection
    canonical = H.select_canonical(registry)

    # Step 5 — Write outputs
    H.write_run_registry(registry, outdir)
    print("Wrote %s/RUN_REGISTRY.csv and .json" % outdir)

    H.write_duplicates_canonical(registry, canonical, outdir)
    print("Wrote DUPLICATES_AND_CANONICAL.md")

    H.write_table_phase3_main(registry, canonical, outdir)
    print("Wrote TABLE_PHASE3_MAIN.md")

    H.write_table_sweep_summary([d for d in sweep_dirs if d.exists()], outdir)
    print("Wrote TABLE_SWEEP_SUMMARY.md")

    H.write_table_policy_behavior(registry, outdir)
    print("Wrote TABLE_POLICY_BEHAVIOR.md")

    cards_dir = outdir / "RUN_CARDS"
    cards_dir.mkdir(parents=True, exist_ok=True)
    for r in registry:
        try:
            H.write_run_card(r, cards_dir)
        except Exception as e:
            print("Run card skip %s: %s" % (r.get("run_id"), e), file=sys.stderr)
    n_cards = len(list(cards_dir.glob("run_*.md")))
    print("Wrote %d run cards to RUN_CARDS/" % n_cards)

    if args.include_figures:
        try:
            H.write_figures(registry, outdir)
            print("Wrote FIGURES/")
        except Exception as e:
            print("Figures skip: %s" % e, file=sys.stderr)

    # Acceptance summary
    n_sweep = sum(1 for r in registry if r.get("run_type") == "sweep")
    n_full = sum(1 for r in registry if r.get("run_type") in ("full_report", "headline"))
    print("Harvest complete: %d sweep, %d full/headline runs; registry and tables in %s" % (n_sweep, n_full, outdir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
