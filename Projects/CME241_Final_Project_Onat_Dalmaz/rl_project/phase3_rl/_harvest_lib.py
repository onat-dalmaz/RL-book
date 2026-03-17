"""
Phase 3 harvest: discover runs, parse artifacts, normalize schema, dedupe, write registry and tables.
Internal library for harvest_results.py.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _v(x: Any, default: str = "—") -> str:
    if x is None or x == "" or (isinstance(x, float) and str(x) == "nan"):
        return default
    return str(x)


def _f(x: Any, default: Optional[float] = None):
    if x is None or x == "" or (isinstance(x, float) and (str(x) == "nan" or x != x)):
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


# --- Bundle extraction ---
def find_bundles(bundles_root: Path, pattern: str = "phase3_snx_results*.tar.gz") -> List[Path]:
    bundles = list(Path(bundles_root).glob(pattern))
    return sorted(bundles, key=lambda p: p.stat().st_mtime, reverse=True)


def extract_bundle(tarball: Path, out_base: Path) -> Path:
    """Extract tarball to out_base / <stem>. Never overwrite existing."""
    stem = tarball.stem
    if stem.endswith(".tar"):
        stem = Path(tarball.name).stem
    dest = out_base / stem
    if dest.exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(dest)
    return dest


def extract_all_bundles(bundles_root: Path, out_base: Path) -> List[Tuple[Path, Path]]:
    """Return list of (tarball_path, extraction_path)."""
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    result = []
    for t in find_bundles(bundles_root):
        try:
            dest = extract_bundle(t, out_base)
            result.append((t, dest))
        except Exception:
            pass
    return result


# --- Run discovery ---
def is_harvestable_dir(d: Path) -> bool:
    return (
        (d / "EVAL_SUMMARY.csv").exists()
        or (d / "RUN_MANIFEST.json").exists()
        or (d / "EFFECTIVE_CONFIG.json").exists()
        or (d / "PASS_FAIL.json").exists()
        or (d / "PASS_FAIL.md").exists()
        or (d / "SWEEP_RESULTS.csv").exists()
    )


def discover_run_dirs(results_root: Path, extracted_base: Optional[Path] = None) -> List[Tuple[Path, str]]:
    """Return [(abs_path, source)] where source is 'active_results' or 'bundle_extracted'."""
    results_root = Path(results_root)
    out = []
    # Active: top-level phase3* dirs and sweep subdirs
    for child in results_root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not (name.startswith("phase3_sweep_snx") or name.startswith("phase3_report_real_snx") or "phase3_report" in name):
            continue
        if is_harvestable_dir(child):
            out.append((child.resolve(), "active_results"))
        if name == "phase3_sweep_snx":
            for sub in child.iterdir():
                if sub.is_dir() and is_harvestable_dir(sub):
                    out.append((sub.resolve(), "active_results"))
    # Extracted bundles: any phase3_* dir under each bundle that is harvestable
    if extracted_base and extracted_base.exists():
        for bundle_dir in extracted_base.iterdir():
            if not bundle_dir.is_dir():
                continue
            for child in bundle_dir.rglob("*"):
                if not child.is_dir():
                    continue
                if child.name.startswith("phase3_") and is_harvestable_dir(child):
                    out.append((child.resolve(), "bundle_extracted"))
    return out


# --- Parsing ---
def parse_effective_config(p: Path) -> Dict[str, Any]:
    if not (p / "EFFECTIVE_CONFIG.json").exists():
        return {}
    try:
        return json.loads((p / "EFFECTIVE_CONFIG.json").read_text())
    except Exception:
        return {}


def parse_run_manifest(p: Path) -> Dict[str, Any]:
    if not (p / "RUN_MANIFEST.json").exists():
        return {}
    try:
        return json.loads((p / "RUN_MANIFEST.json").read_text())
    except Exception:
        return {}


def parse_dp_phase2_model(p: Path) -> Dict[str, Any]:
    if not (p / "DP_PHASE2_MODEL.json").exists():
        return {}
    try:
        return json.loads((p / "DP_PHASE2_MODEL.json").read_text())
    except Exception:
        return {}


def parse_pass_fail(p: Path) -> Tuple[str, Dict]:
    for name in ["PASS_FAIL.json", "PASS_FAIL.md"]:
        f = p / name
        if f.exists():
            try:
                text = f.read_text()
                if name.endswith(".json"):
                    return (json.loads(text).get("overall", "NA"), json.loads(text) if text else {})
                return ("PASS" if "PASS" in text.upper() else "FAIL", {})
            except Exception:
                pass
    return ("NA", {})


def infer_config_from_path(path: Path) -> Dict[str, Any]:
    name = path.name
    out = {}
    m = re.search(r"cfg(\d+)", name, re.I)
    if m:
        out["config_id"] = int(m.group(1))
    m = re.search(r"cm([\d.]+)", name, re.I)
    if m:
        out["c_maker_bps"] = float(m.group(1))
    m = re.search(r"p0([\d.]+)", name, re.I)
    if m:
        out["p0"] = float(m.group(1))
    m = re.search(r"p1([\d.]+)", name, re.I)
    if m:
        out["p1"] = float(m.group(1))
    return out


def parse_eval_summary(p: Path) -> Dict[str, Any]:
    """Extract QL row and key baseline rows from EVAL_SUMMARY.csv. Return flat dict for registry."""
    fp = p / "EVAL_SUMMARY.csv"
    if not fp.exists():
        return {}
    try:
        df = pd.read_csv(fp)
    except Exception:
        return {}
    if "policy" not in df.columns:
        return {}
    out = {}
    ql = df[df["policy"] == "QL"]
    if not ql.empty:
        q = ql.iloc[0]
        out["ql_mean"] = q.get("mean_cum_bps")
    for k in ["ci_low", "ci_high"]:
        if k in q:
            out["ql_" + k] = q[k]
    for k in ["best_fair_baseline", "delta_fair_mean", "delta_fair_ci_low", "delta_fair_ci_high",
              "gap_to_oracle_mean", "gap_to_oracle_ci_low", "gap_to_oracle_ci_high", "maker_share", "taker_share"]:
        if k in q:
            out[k] = q[k]
    for pol in ["DP_EXACT", "DP_PHASE2", "A_sign_taker", "B_sign_maker", "Hold"]:
        row = df[df["policy"] == pol]
        if not row.empty and "mean_cum_bps" in row.columns:
            out[f"{pol.lower().replace('_', '')}_mean"] = row.iloc[0]["mean_cum_bps"]
    return out


def parse_eval_visitation(p: Path) -> Dict[str, Any]:
    fp = p / "EVAL_VISITATION.csv"
    if not fp.exists():
        return {}
    try:
        df = pd.read_csv(fp)
        out = {}
        if "maker_share" in df.columns:
            out["maker_share_mean"] = float(df["maker_share"].mean())
        if "taker_share" in df.columns:
            out["taker_share_mean"] = float(df["taker_share"].mean())
        if "action_frac_HOLD" in df.columns:
            out["hold_frac_mean"] = float(df["action_frac_HOLD"].mean())
        if "turnover_pct" in df.columns:
            out["turnover_mean"] = float(df["turnover_pct"].mean())
        if "avg_abs_inv" in df.columns:
            out["avg_abs_inv_mean"] = float(df["avg_abs_inv"].mean())
        return out
    except Exception:
        return {}


def run_id_from_path(path: Path, manifest_ts: Optional[str] = None) -> str:
    h = hashlib.sha256(str(path).encode()).hexdigest()[:12]
    ts = (manifest_ts or "")[:19].replace(":", "").replace("-", "").replace("T", "_")
    return f"{path.name}_{ts}_{h}" if ts else f"{path.name}_{h}"


def normalize_run(path: Path, source: str) -> Dict[str, Any]:
    """Build one registry row for a run directory."""
    path = Path(path)
    eff = parse_effective_config(path)
    manifest = parse_run_manifest(path)
    dp2 = parse_dp_phase2_model(path)
    pass_overall, _ = parse_pass_fail(path)
    eval_metrics = parse_eval_summary(path)
    vis = parse_eval_visitation(path)
    from_path = infer_config_from_path(path)

    exec_c = eff.get("execution", {}) or manifest.get("args", {}) or {}
    train_c = eff.get("training", {}) or manifest.get("args", {}) or {}
    wind_c = eff.get("windowing", {}) or manifest.get("args", {}) or {}
    boot_c = eff.get("bootstrap", {}) or manifest.get("args", {}) or {}

    def get(key: str, *sources: Dict) -> Any:
        for d in sources:
            if d and key in d:
                return d[key]
        return None

    created_ts = manifest.get("ts") or get("ts", manifest)
    run_id = run_id_from_path(path, created_ts)

    config_id = from_path.get("config_id") or get("config_id", exec_c, manifest.get("args", {}))
    c_maker = _f(get("c_maker_bps", exec_c, manifest.get("args", {})) or from_path.get("c_maker_bps"))
    c_taker = _f(get("c_taker_bps", exec_c, manifest.get("args", {})))
    p0 = _f(get("p0", exec_c, manifest.get("args", {})) or from_path.get("p0"))
    p1 = _f(get("p1", exec_c, manifest.get("args", {})) or from_path.get("p1"))

    has_qt = (path / "Q_TABLE_QL.npy").exists() or (path / "Q_TABLE.csv").exists()
    has_pol = (path / "POLICY_TABLE_QL.csv").exists() or (path / "POLICY_TABLE.csv").exists()
    has_eval = (path / "EVAL_SUMMARY.csv").exists()
    is_complete = bool(has_eval and (has_qt and has_pol or "sweep" in path.name.lower()))
    if "sweep" in path.name and (path / "EVAL_SUMMARY.csv").exists():
        is_complete = True

    run_type = "sweep" if "SWEEP_RESULTS.csv" in [f.name for f in path.iterdir() if f.is_file()] else "full_report"
    if "HEADLINE" in path.name:
        run_type = "headline"

    schema_version = "v2" if (has_eval and eval_metrics and "delta_fair_mean" in eval_metrics) else "v1"

    return {
        "run_id": run_id,
        "run_type": run_type,
        "source": source,
        "path": str(path),
        "name": path.name,
        "created_ts": _v(created_ts),
        "git_hash": _v(manifest.get("git_hash")),
        "config_id": _v(config_id),
        "c_maker": c_maker,
        "c_taker": c_taker,
        "p0": p0,
        "p1": p1,
        "dv": _f(get("dv", exec_c, manifest.get("args", {}))),
        "d_age": _f(get("d_age", exec_c, manifest.get("args", {}))),
        "reward_mode": _v(get("reward_mode", eff.get("env", {}), manifest.get("args", {}))),
        "eta_turnover": _f(get("eta_turnover", exec_c, manifest.get("args", {}))),
        "lambda_inv": _f(get("lambda_inv", exec_c, manifest.get("args", {}))),
        "episodes": get("n_train_episodes", train_c, manifest.get("args", {})),
        "train_windows": get("n_train_windows", wind_c, manifest.get("args", {})),
        "eval_windows": get("eval_num_windows", wind_c, manifest.get("args", {})),
        "fill_seeds": get("eval_fill_seeds", wind_c, manifest.get("args", {})),
        "bootstrap_iters": get("bootstrap_iters", boot_c, manifest.get("args", {})),
        "schema_version": schema_version,
        "is_complete": is_complete,
        "has_dp_exact": (path / "DP_POLICY_TABLE.csv").exists() or (path / "DP_SUMMARY.csv").exists(),
        "has_dp_phase2": (path / "DP_PHASE2_MODEL.json").exists(),
        "dp_phase2_reward_mode": _v(dp2.get("reward_mode", "drift")),
        "has_dp_empirical": (path / "DP_EMPIRICAL_POLICY_TABLE.csv").exists(),
        "pass_fail_overall": _v(pass_overall),
        "notes": "",
        "invalid_run": (path / "INVALID_RUN.txt").exists(),
        **{k: v for k, v in eval_metrics.items()},
        **{k: v for k, v in vis.items()},
    }


# --- Dedupe and canonical ---
def group_key(r: Dict[str, Any]) -> Tuple:
    return (
        _v(r.get("config_id")),
        _f(r.get("c_maker")),
        _f(r.get("p0")),
        _f(r.get("p1")),
        r.get("eval_windows"),
        r.get("fill_seeds"),
        r.get("bootstrap_iters"),
        _v(r.get("dp_phase2_reward_mode")),
    )


def rank_for_canonical(r: Dict[str, Any]) -> Tuple:
    pass_order = {"PASS": 0, "WARN": 1, "FAIL": 2, "NA": 3}
    po = pass_order.get(str(r.get("pass_fail_overall", "NA")), 3)
    has_bundle = 0 if (Path(r["path"]) / "phase3_bundle.zip").exists() else 1
    ts = (r.get("created_ts") or "")[:19]
    return (po, has_bundle, ts)


def select_canonical(registry: List[Dict[str, Any]]) -> Dict[Tuple, Dict[str, Any]]:
    groups: Dict[Tuple, List[Dict]] = {}
    for r in registry:
        if r.get("run_type") == "sweep":
            continue
        k = group_key(r)
        groups.setdefault(k, []).append(r)
    canonical = {}
    for k, runs in groups.items():
        runs_sorted = sorted(runs, key=rank_for_canonical)
        canonical[k] = runs_sorted[0] if runs_sorted else None
    return canonical


# --- Writers ---
REGISTRY_COLUMNS = [
    "run_id", "run_type", "source", "path", "name", "created_ts", "git_hash",
    "config_id", "c_maker", "c_taker", "p0", "p1", "dv", "d_age",
    "reward_mode", "eta_turnover", "lambda_inv",
    "episodes", "train_windows", "eval_windows", "fill_seeds", "bootstrap_iters",
    "schema_version", "is_complete", "has_dp_exact", "has_dp_phase2", "dp_phase2_reward_mode", "has_dp_empirical",
    "pass_fail_overall", "notes", "invalid_run",
    "ql_mean", "ql_ci_low", "ql_ci_high", "best_fair_baseline", "delta_fair_mean", "delta_fair_ci_low", "delta_fair_ci_high",
    "gap_to_oracle_mean", "gap_to_oracle_ci_low", "gap_to_oracle_ci_high",
    "dpexact_mean", "dpphase2_mean", "maker_share_mean", "taker_share_mean", "hold_frac_mean", "turnover_mean", "avg_abs_inv_mean",
]


def flatten_for_csv(r: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for c in REGISTRY_COLUMNS:
        out[c] = r.get(c, "")
    for k, v in r.items():
        if k not in out and not k.startswith("_"):
            out[k] = v
    return out


def write_run_registry(registry: List[Dict[str, Any]], outdir: Path) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [flatten_for_csv(r) for r in registry]
    df = pd.DataFrame(rows)
    cols = [c for c in REGISTRY_COLUMNS if c in df.columns]
    df = df.reindex(columns=cols + [c for c in df.columns if c not in cols])
    df.to_csv(outdir / "RUN_REGISTRY.csv", index=False)
    with open(outdir / "RUN_REGISTRY.json", "w") as f:
        json.dump(registry, f, indent=2, default=str)


def write_duplicates_canonical(registry: List[Dict], canonical: Dict[Tuple, Optional[Dict]], outdir: Path) -> None:
    outdir = Path(outdir)
    lines = ["# Duplicates and canonical selection", ""]
    groups: Dict[Tuple, List[Dict]] = {}
    for r in registry:
        if r.get("run_type") == "sweep":
            continue
        k = group_key(r)
        groups.setdefault(k, []).append(r)
    for k, runs in groups.items():
        can = canonical.get(k)
        lines.append(f"## Group {k}")
        lines.append(f"Canonical: {can['name']} ({can['path']})" if can else "None")
        for r in runs:
            mark = " **CANONICAL**" if can and r["run_id"] == can["run_id"] else ""
            lines.append(f"- {r['name']} ({r['source']}){mark}")
        lines.append("")
    (outdir / "DUPLICATES_AND_CANONICAL.md").write_text("\n".join(lines))


def write_table_phase3_main(registry: List[Dict], canonical: Dict[Tuple, Optional[Dict]], outdir: Path) -> None:
    outdir = Path(outdir)
    # Dedupe by path, then by run name (one row per logical run; prefer zero_mean for headline)
    by_path: Dict[str, Dict] = {}
    for r in registry:
        if r.get("run_type") == "sweep":
            continue
        path = str(r.get("path", ""))
        if not path:
            continue
        if path not in by_path or r.get("source") == "active_results":
            by_path[path] = r
    by_name: Dict[str, Dict] = {}
    for r in by_path.values():
        name = r.get("name", "")
        if not name:
            continue
        if name not in by_name:
            by_name[name] = r
        else:
            # Prefer zero_mean for same run name (headline with fixed DP_PHASE2)
            if r.get("dp_phase2_reward_mode") == "zero_mean" and by_name[name].get("dp_phase2_reward_mode") != "zero_mean":
                by_name[name] = r
            elif r.get("source") == "active_results" and by_name[name].get("source") != "active_results":
                by_name[name] = r
    # Include headline first, then cfg2, cfg6
    priority = []
    for r in by_name.values():
        name = r.get("name", "")
        if "HEADLINE" in name:
            priority.append((0, r))
        elif "cfg2" in name:
            priority.append((1, r))
        elif "cfg6" in name:
            priority.append((2, r))
        else:
            priority.append((3, r))
    priority.sort(key=lambda x: (x[0], x[1]["name"]))
    seen_paths = set()
    rows = []
    for _, r in priority:
        path = r.get("path", "")
        if path in seen_paths:
            continue
        seen_paths.add(path)
        if not r.get("is_complete"):
            continue
        ql_mean = _v(r.get("ql_mean"), "—")
        ql_ci = f"[{_v(r.get('ql_ci_low'), '—')}, {_v(r.get('ql_ci_high'), '—')}]"
        best_fair = _v(r.get("best_fair_baseline"), "—")
        df_mean = _v(r.get("delta_fair_mean"), "—")
        df_ci = f"[{_v(r.get('delta_fair_ci_low'), '—')}, {_v(r.get('delta_fair_ci_high'), '—')}]"
        gap_mean = _v(r.get("gap_to_oracle_mean"), "—")
        gap_ci = f"[{_v(r.get('gap_to_oracle_ci_low'), '—')}, {_v(r.get('gap_to_oracle_ci_high'), '—')}]"
        dp2_mean = _v(r.get("dpphase2_mean"), "—")
        dpe_mean = _v(r.get("dpexact_mean"), "—")
        ms = _v(r.get("maker_share_mean") or r.get("maker_share"), "—")
        ts = _v(r.get("taker_share_mean") or r.get("taker_share"), "—")
        hf = _v(r.get("hold_frac_mean"), "—")
        turn = _v(r.get("turnover_mean"), "—")
        inv = _v(r.get("avg_abs_inv_mean"), "—")
        dp2_mode = _v(r.get("dp_phase2_reward_mode"), "—")
        rows.append(f"| {r['name']} | {ql_mean} | {ql_ci} | {best_fair} | {df_mean} | {df_ci} | {dpe_mean} | {gap_mean} | {gap_ci} | {ms} | {ts} | {hf} | {turn} | {inv} | {dp2_mode} |")
    header = "| Run | QL mean | QL CI | Best fair | Delta_fair mean | Delta_fair CI | DP_EXACT mean | Gap_to_oracle mean | Gap CI | maker_share | taker_share | hold_frac | turnover | avg_abs_inv | dp_phase2_reward_mode |"
    sep = "|" + "---|" * (header.count("|") - 1)
    (outdir / "TABLE_PHASE3_MAIN.md").write_text("# Phase 3 main metrics\n\n" + header + "\n" + sep + "\n" + "\n".join(rows[:20]) + "\n")


def write_table_sweep_summary(sweep_dirs: List[Path], outdir: Path) -> None:
    outdir = Path(outdir)
    lines = ["# Sweep summary (top configs)", ""]
    for d in sweep_dirs:
        fp = d / "SWEEP_RESULTS.csv"
        if not fp.exists():
            fp = d / "sweep_0" / "EVAL_SUMMARY.csv"
        if fp.exists() and "SWEEP" in str(fp):
            try:
                df = pd.read_csv(fp)
                lines.append(f"## {d.name}")
                lines.append(df.head(10).to_markdown(index=False) if hasattr(df, "to_markdown") else df.head(10).to_string())
                lines.append("")
            except Exception:
                pass
        top = d / "TOP_CONFIGS.json"
        if top.exists():
            try:
                cfg = json.loads(top.read_text())
                lines.append(f"TOP_CONFIGS: {cfg[:5]}")
                lines.append("")
            except Exception:
                pass
    (outdir / "TABLE_SWEEP_SUMMARY.md").write_text("\n".join(lines) if lines else "# No sweep data\n")


def write_table_policy_behavior(registry: List[Dict], outdir: Path) -> None:
    outdir = Path(outdir)
    lines = ["# Policy behavior", "", "| Run | maker_share | taker_share | hold_frac | turnover | avg_abs_inv |", "|-----|-------------|--------------|-----------|----------|-------------|"]
    for r in registry:
        if r.get("run_type") == "sweep":
            continue
        if not r.get("is_complete"):
            continue
        lines.append(f"| {r['name']} | {_v(r.get('maker_share_mean') or r.get('maker_share'))} | {_v(r.get('taker_share_mean') or r.get('taker_share'))} | {_v(r.get('hold_frac_mean'))} | {_v(r.get('turnover_mean'))} | {_v(r.get('avg_abs_inv_mean'))} |")
    (outdir / "TABLE_POLICY_BEHAVIOR.md").write_text("\n".join(lines[:25]) + "\n")


def write_run_card(r: Dict[str, Any], outdir: Path) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rid = (r.get("run_id") or "unknown").replace("/", "_")[:80]
    path = Path(r["path"])
    lines = [
        f"# Run card: {r.get('name', rid)}",
        "",
        f"- **run_id:** {r.get('run_id')}",
        f"- **path:** {r['path']}",
        f"- **source:** {r.get('source')}",
        f"- **created_ts:** {r.get('created_ts')}",
        "",
        "## Key settings",
        f"- c_maker={r.get('c_maker')}, c_taker={r.get('c_taker')}, p0={r.get('p0')}, p1={r.get('p1')}",
        f"- eta_turnover={r.get('eta_turnover')}, lambda_inv={r.get('lambda_inv')}",
        f"- eval_windows={r.get('eval_windows')}, fill_seeds={r.get('fill_seeds')}, bootstrap_iters={r.get('bootstrap_iters')}",
        f"- dp_phase2_reward_mode={r.get('dp_phase2_reward_mode')}",
        "",
        "## Metrics",
        f"- QL mean: {r.get('ql_mean')} CI [{r.get('ql_ci_low')}, {r.get('ql_ci_high')}]",
        f"- best_fair_baseline: {r.get('best_fair_baseline')}",
        f"- delta_fair_mean: {r.get('delta_fair_mean')} CI [{r.get('delta_fair_ci_low')}, {r.get('delta_fair_ci_high')}]",
        f"- gap_to_oracle_mean: {r.get('gap_to_oracle_mean')}",
        "",
        "## File pointers",
        f"- EVAL_SUMMARY.csv: `{path / 'EVAL_SUMMARY.csv'}`",
        f"- DP_PHASE2_MODEL.json: `{path / 'DP_PHASE2_MODEL.json'}`" if (path / "DP_PHASE2_MODEL.json").exists() else "- DP_PHASE2_MODEL.json: N/A",
        f"- POLICY_TABLE_QL.csv: `{path / 'POLICY_TABLE_QL.csv'}`" if (path / "POLICY_TABLE_QL.csv").exists() else "- POLICY_TABLE_QL.csv: N/A",
        f"- phase3_bundle.zip: `{path / 'phase3_bundle.zip'}`" if (path / "phase3_bundle.zip").exists() else "- phase3_bundle.zip: N/A",
        "",
        "## OK to claim / Don't claim",
        "- Delta_fair mean > 0 and CI lower > 0: can claim 'statistically significant improvement over fair baseline'.",
        "- Delta_fair mean > 0, CI crosses 0: report as 'positive point estimate'.",
    ]
    (outdir / f"run_{rid}.md").write_text("\n".join(lines))


def write_figures(registry: List[Dict], outdir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    outdir = Path(outdir) / "FIGURES"
    outdir.mkdir(parents=True, exist_ok=True)
    runs = [r for r in registry if r.get("run_type") != "sweep" and r.get("is_complete") and r.get("ql_mean") is not None][:15]
    if not runs:
        return
    names = [r["name"][:20] for r in runs]
    df_means = [_f(r.get("delta_fair_mean")) or 0 for r in runs]
    df_lo = [_f(r.get("delta_fair_ci_low")) or 0 for r in runs]
    df_hi = [_f(r.get("delta_fair_ci_high")) or 0 for r in runs]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(names))
    ax.bar(x, df_means, yerr=[[df_means[i] - (df_lo[i] or df_means[i]) for i in range(len(df_means))], [(df_hi[i] or df_means[i]) - df_means[i] for i in range(len(df_means))]], capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Delta_fair mean (bps)")
    ax.axhline(0, color="gray", linestyle="--")
    plt.tight_layout()
    plt.savefig(outdir / "delta_fair_bar.png", dpi=100)
    plt.close()
