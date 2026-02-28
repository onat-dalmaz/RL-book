"""
Phase 2 DP v3: Discovery, schema classification, canonicalization, z construction,
transition model P(z'|z), and reward model (Step1 counterfactual / S7 empirical).
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

# Step1: ts, y, pred (required); pos optional
STEP1_TS = ["ts", "timestamp"]
STEP1_Y = ["y"]
STEP1_PRED = ["pred"]
# S7: entry_ts, side, net_bps (required); pred optional
S7_TS = ["entry_ts", "entry_time"]
S7_SIDE = ["side"]
S7_NET_BPS = ["net_bps", "net_pnl_bps", "net_exec_bps"]
S7_PRED = ["pred_at_entry", "pred", "signal"]


def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def classify_schema(df: pd.DataFrame) -> str:
    """Classify as step1, s7, or unknown."""
    if df is None or df.empty:
        return "unknown"
    cols = set(df.columns)
    has_ts = _find_col(df, STEP1_TS) is not None
    has_y = _find_col(df, STEP1_Y) is not None
    has_pred_s1 = _find_col(df, STEP1_PRED) is not None
    has_entry_ts = _find_col(df, S7_TS) is not None
    has_side = _find_col(df, S7_SIDE) is not None
    has_net_bps = _find_col(df, S7_NET_BPS) is not None

    if has_ts and has_y and has_pred_s1:
        return "step1"
    if has_entry_ts and has_side and has_net_bps:
        return "s7"
    return "unknown"


def parse_path_metadata(path: Path, run_root: Path) -> tuple:
    """Return (coin, variant, fold, split). Prefer COINS/<coin>/cv/<variant>/fold_<k>/."""
    try:
        rel = path.relative_to(run_root)
    except ValueError:
        rel = path
    parts = rel.parts
    coin = variant = fold = split = None
    # COINS/<coin>/... (when run_root is above COINS)
    if len(parts) >= 2 and parts[0] == "COINS":
        coin = parts[1]
    # .../cv/<variant>/fold_<k>/... (e.g. cv/SNX/fold_0/... under run_root=COINS/SNX)
    for i, p in enumerate(parts):
        if p.startswith("fold_"):
            try:
                fold = int(p.split("_")[1])
            except (IndexError, ValueError):
                fold = 0
            if i >= 1:
                variant = parts[i - 1]
            break
    if variant is None and len(parts) >= 2 and parts[0] == "cv":
        variant = parts[1]
        if coin is None:
            coin = parts[1]
    if coin is None and len(parts) >= 2 and parts[0] == "cv":
        coin = parts[1]
    if coin is None and len(parts) >= 1:
        coin = parts[0]
    if variant is None:
        variant = coin or "UNK"
    if "test" in path.name.lower() or "trades_test" in path.name:
        split = "test"
    elif "val" in path.name.lower() or "trades_val" in path.name:
        split = "val"
    elif "train" in path.name.lower() or "trades_train" in path.name:
        split = "train"
    return coin or "UNK", variant or "UNK", fold if fold is not None else 0, split or "test"


# --- Phase 0: Discovery ---

STEP1_NAMES = [
    "trades_train.parquet", "trades_val.parquet", "trades_test.parquet",
    "train_trades.parquet", "val_trades.parquet", "test_trades.parquet",
]
S7_NAMES = [
    "train_trades.parquet", "val_trades.parquet", "test_trades.parquet",
]


def discover_and_classify(run_root: Path, mode: str, outdir: Path):
    """
    Scan run_root for parquets; classify each as step1|s7|unknown.
    mode: auto (accept both), step1 (only step1), s7 (only s7).
    Returns (manifest_df, missing_rows).
    """
    run_root = Path(run_root)
    outdir = Path(outdir)
    all_names = list(dict.fromkeys(STEP1_NAMES + S7_NAMES))
    rows = []
    missing_rows = []
    dataset_id_by_key = {}

    for name in all_names:
        for path in run_root.rglob(name):
            if not path.is_file() or pq is None:
                continue
            try:
                t = pq.read_table(str(path))
                df = t.to_pandas()
                n = len(df)
                kind = classify_schema(df)
                if mode == "step1" and kind != "step1":
                    continue
                if mode == "s7" and kind != "s7":
                    continue
                if kind == "unknown":
                    missing_rows.append({
                        "path": str(path.relative_to(run_root)),
                        "reason": "UNKNOWN_SCHEMA",
                        "columns": json.dumps(list(df.columns)),
                    })
                    continue

                ts_col = _find_col(df, STEP1_TS if kind == "step1" else S7_TS)
                min_ts = max_ts = None
                if ts_col:
                    ts = pd.to_datetime(df[ts_col], errors="coerce").dropna()
                    if len(ts) > 0:
                        min_ts = str(ts.min())
                        max_ts = str(ts.max())

                coin, variant, fold, split = parse_path_metadata(path, run_root)
                key = (coin, variant, fold, split)
                dataset_id = f"{coin}_{variant}_fold{fold}_{split}"
                if key not in dataset_id_by_key:
                    dataset_id_by_key[key] = dataset_id

                rows.append({
                    "dataset_id": dataset_id,
                    "coin": coin,
                    "variant": variant,
                    "fold": fold,
                    "split": split,
                    "kind": kind,
                    "path": str(path.relative_to(run_root)),
                    "n_rows": n,
                    "min_ts": min_ts,
                    "max_ts": max_ts,
                    "cols_json": json.dumps(list(df.columns)),
                })
            except Exception as e:
                missing_rows.append({
                    "path": str(path.relative_to(run_root)),
                    "reason": str(e)[:200],
                    "columns": "",
                })

    if not rows:
        pd.DataFrame(columns=[
            "dataset_id", "coin", "variant", "fold", "split", "kind", "path",
            "n_rows", "min_ts", "max_ts", "cols_json"
        ]).to_csv(outdir / "PHASE2_DP_MANIFEST.csv", index=False)
        if missing_rows:
            pd.DataFrame(missing_rows).to_csv(outdir / "MISSING_COLUMNS.csv", index=False)
        return pd.DataFrame(), missing_rows

    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(outdir / "PHASE2_DP_MANIFEST.csv", index=False)
    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(outdir / "MISSING_COLUMNS.csv", index=False)
    return manifest_df, missing_rows


# --- Phase 1: Canonicalization ---

def load_and_canonicalize(path: Path, run_root: Path, kind: str, allow_no_pred_s7: bool = False):
    """
    Load parquet and return canonical df.
    Step1: columns t, pred, y (and optionally pos).
    S7: columns t, net_bps, side; pred if present or allow_no_pred_s7.
    Returns (df, error_msg). error_msg None on success.
    """
    path = Path(path)
    run_root = Path(run_root)
    full = path if path.is_absolute() or path.exists() else run_root / path
    if not full.exists() or pq is None:
        return None, "path_missing_or_no_pyarrow"
    df = pq.read_table(str(full)).to_pandas()
    if df.empty:
        return None, "no_rows"

    if kind == "step1":
        ts_c = _find_col(df, STEP1_TS)
        y_c = _find_col(df, STEP1_Y)
        pred_c = _find_col(df, STEP1_PRED)
        if not ts_c or not y_c or not pred_c:
            return None, "step1_missing_ts_y_pred"
        df = df.copy()
        df["t"] = pd.to_datetime(df[ts_c], utc=True)
        df["pred"] = pd.to_numeric(df[pred_c], errors="coerce").fillna(0)
        df["y"] = pd.to_numeric(df[y_c], errors="coerce").fillna(0)
        df = df.sort_values("t").reset_index(drop=True)
        out = df[["t", "pred", "y"]].copy()
        return out, None

    # S7
    ts_c = _find_col(df, S7_TS)
    side_c = _find_col(df, S7_SIDE)
    net_c = _find_col(df, S7_NET_BPS)
    pred_c = _find_col(df, S7_PRED)
    if not ts_c or not side_c or not net_c:
        return None, "s7_missing_entry_ts_side_net_bps"
    if not pred_c and not allow_no_pred_s7:
        return None, "s7_missing_pred"
    df = df.copy()
    df["t"] = pd.to_datetime(df[ts_c], utc=True)
    df["net_bps"] = pd.to_numeric(df[net_c], errors="coerce").fillna(0)
    raw = df[side_c]
    if raw.dtype == object or raw.dtype.name == "string":
        df["side"] = np.where(raw.str.lower().str.contains("long|buy|1").fillna(False), 1, -1)
    else:
        vals = pd.to_numeric(raw, errors="coerce").fillna(0)
        if vals.max() <= 1 and vals.min() >= 0:
            df["side"] = np.where(vals > 0, 1, -1)
        else:
            df["side"] = np.sign(vals).replace(0, 1)
    if pred_c:
        df["pred"] = pd.to_numeric(df[pred_c], errors="coerce").fillna(0)
    else:
        df["pred"] = 0.0
    df = df.sort_values("t").reset_index(drop=True)
    out = df[["t", "pred", "net_bps", "side"]].copy()
    return out, None


def canonicalize_all(manifest_df: pd.DataFrame, run_root: Path, outdir: Path, allow_no_pred_s7: bool):
    """Write CANONICAL_PREVIEW_step1.csv and CANONICAL_PREVIEW_s7.csv (head 100)."""
    run_root = Path(run_root)
    outdir = Path(outdir)
    step1_dfs = []
    s7_dfs = []
    for _, r in manifest_df.iterrows():
        path = run_root / r["path"]
        kind = r["kind"]
        df, err = load_and_canonicalize(path, run_root, kind, allow_no_pred_s7)
        if err or df is None:
            continue
        df = df.head(100)
        df["dataset_id"] = r["dataset_id"]
        if kind == "step1":
            step1_dfs.append(df)
        else:
            s7_dfs.append(df)
    if step1_dfs:
        pd.concat(step1_dfs, ignore_index=True).to_csv(outdir / "CANONICAL_PREVIEW_step1.csv", index=False)
    if s7_dfs:
        pd.concat(s7_dfs, ignore_index=True).to_csv(outdir / "CANONICAL_PREVIEW_s7.csv", index=False)
    return True


# --- Phase 2 v4: Horizon ---

def infer_label_horizon_ms(df, run_root: Path):
    """Returns (raw_dt_ms, label_horizon_ms, horizon_source)."""
    run_root = Path(run_root)
    raw_dt_ms = 500.0
    if df is not None and len(df) > 1 and "t" in df.columns:
        ts = pd.to_datetime(df["t"], utc=True)
        diff_ms = ts.diff().dropna().dt.total_seconds() * 1000
        raw_dt_ms = float(diff_ms.median())
        if raw_dt_ms <= 0 or not np.isfinite(raw_dt_ms):
            raw_dt_ms = 500.0
    label_horizon_ms = raw_dt_ms
    horizon_source = "unknown"
    for p in [run_root / "RUN_MANIFEST.json"] + list(run_root.rglob("RUN_MANIFEST.json"))[:1]:
        if p.exists():
            try:
                d = json.loads(p.read_text())
                if "horizon_ms" in d:
                    label_horizon_ms = float(d["horizon_ms"])
                    horizon_source = "manifest"
                    break
                if "horizon_sec" in d:
                    label_horizon_ms = float(d["horizon_sec"]) * 1000
                    horizon_source = "manifest"
                    break
            except Exception:
                pass
            break
    if horizon_source == "unknown":
        label_horizon_ms = raw_dt_ms
    return (raw_dt_ms, label_horizon_ms, horizon_source)


def write_horizon_spec(outdir: Path, raw_dt_ms: float, label_horizon_ms: float, stride: int,
                      mdp_step_ms: float, stride_mode: str, horizon_source: str):
    spec = {"raw_dt_ms": raw_dt_ms, "label_horizon_ms": label_horizon_ms, "stride": stride,
            "mdp_step_ms": mdp_step_ms, "stride_mode": stride_mode, "horizon_source": horizon_source}
    (outdir / "HORIZON_SPEC.json").write_text(json.dumps(spec, indent=2))


# --- Phase 2: Z thresholds and distribution ---

def select_threshold_fit_df(dfs_by_split: dict, kind: str, z_fit_mode: str = "auto") -> tuple:
    """
    Returns (fit_split_name, fit_df).
    z_fit_mode: auto | train | val | val_and_test | test70
    - auto: train > val > first 70%% test (no leakage).
    - train/val/test70: use that split if present.
    - val_and_test: concatenate val+test and fit on that (leakage mode).
    """
    if not dfs_by_split:
        return None, None
    val_df = dfs_by_split.get("val")
    test_df = dfs_by_split.get("test")
    if z_fit_mode == "val_and_test":
        if val_df is not None and test_df is not None and len(val_df) > 0 and len(test_df) > 0:
            return "val_and_test", pd.concat([val_df, test_df], ignore_index=True)
        return None, None
    if z_fit_mode == "train":
        if dfs_by_split.get("train") is not None and len(dfs_by_split["train"]) > 0:
            return "train", dfs_by_split["train"]
        return None, None
    if z_fit_mode == "val":
        if val_df is not None and len(val_df) > 0:
            return "val", val_df
        return None, None
    if z_fit_mode == "test70":
        if test_df is not None and len(test_df) > 0:
            n70 = max(1, int(0.7 * len(test_df)))
            return "test70", test_df.iloc[:n70]
        return None, None
    # auto: original priority
    if dfs_by_split.get("train") is not None and len(dfs_by_split["train"]) > 0:
        return "train", dfs_by_split["train"]
    if val_df is not None and len(val_df) > 0:
        return "val", val_df
    if test_df is not None and len(test_df) > 0:
        n70 = max(1, int(0.7 * len(test_df)))
        return "test70", test_df.iloc[:n70]
    return None, None


def compute_z_thresholds(manifest_df: pd.DataFrame, run_root: Path, q_lo: float, q_hi: float, outdir: Path,
                        min_z_frac: float = 0.05, z_fit_mode: str = "auto"):
    """
    Per dataset_id: fit thresholds per z_fit_mode (auto | train | val | val_and_test | test70).
    """
    run_root = Path(run_root)
    z_thresholds = []
    z_distribution = []
    gate_errors = []

    for key, grp in manifest_df.groupby(["coin", "variant", "fold"]):
        coin, variant, fold = key
        dataset_id = f"{coin}_{variant}_fold{fold}"
        dfs_by_split = {}
        n_train = n_val = n_test = 0
        for _, r in grp.iterrows():
            path = run_root / r["path"]
            if not path.exists():
                continue
            df, err = load_and_canonicalize(path, run_root, r["kind"], allow_no_pred_s7=True)
            if err or df is None or len(df) == 0:
                continue
            s = r["split"]
            dfs_by_split[s] = df
            if s == "train":
                n_train = len(df)
            elif s == "val":
                n_val = len(df)
            elif s == "test":
                n_test = len(df)

        fit_split_name, fit_df = select_threshold_fit_df(dfs_by_split, grp.iloc[0]["kind"], z_fit_mode=z_fit_mode)
        if fit_df is None or len(fit_df) == 0:
            gate_errors.append((dataset_id, "no_fit_split", ""))
            continue

        fit_pred = fit_df["pred"].values
        fit_start_ts = str(fit_df["t"].min()) if "t" in fit_df.columns else ""
        fit_end_ts = str(fit_df["t"].max()) if "t" in fit_df.columns else ""
        n_fit = len(fit_pred)
        pred_std = float(np.nanstd(fit_pred))
        pred_mean = float(np.nanmean(fit_pred))
        q_lo_val = np.nanpercentile(fit_pred, q_lo * 100)
        q_hi_val = np.nanpercentile(fit_pred, q_hi * 100)
        if np.isnan(q_lo_val):
            q_lo_val = -0.01
        if np.isnan(q_hi_val):
            q_hi_val = 0.01

        if pred_std <= 0:
            gate_errors.append((dataset_id, "pred_std_leq_0", str(pred_std)))
        if q_hi_val <= q_lo_val:
            gate_errors.append((dataset_id, "q_hi_leq_q_lo", ""))

        if fit_split_name != "val_and_test":
            if fit_split_name == "train" and n_train > 0 and n_fit != n_train:
                gate_errors.append((dataset_id, "z_threshold_no_test_leak", f"n_fit={n_fit} != n_train={n_train}"))
            elif fit_split_name == "val" and n_val > 0 and n_fit != n_val:
                gate_errors.append((dataset_id, "z_threshold_no_test_leak", f"n_fit={n_fit} != n_val={n_val}"))
            elif fit_split_name == "test70" and n_test > 0 and n_fit > n_test:
                gate_errors.append((dataset_id, "z_threshold_no_test_leak", "test70 > n_test"))

        z_thresholds.append({
            "dataset_id": dataset_id,
            "coin": coin, "variant": variant, "fold": fold,
            "q_lo": q_lo_val, "q_hi": q_hi_val,
            "pred_mean": pred_mean, "pred_std": pred_std,
            "n_fit": n_fit,
            "fit_split": fit_split_name,
            "fit_start_ts": fit_start_ts,
            "fit_end_ts": fit_end_ts,
        })

        # Z distribution on each split
        for _, r in grp.iterrows():
            path = run_root / r["path"]
            kind = r["kind"]
            df, _ = load_and_canonicalize(path, run_root, kind, allow_no_pred_s7=True)
            if df is None or len(df) == 0:
                continue
            pred = df["pred"].values
            z = np.zeros(len(pred), dtype=int)
            z[pred >= q_hi_val] = 1
            z[pred <= q_lo_val] = -1
            for zi in (-1, 0, 1):
                count = int((z == zi).sum())
                frac = count / max(1, len(z))
                z_distribution.append({
                    "dataset_id": dataset_id,
                    "split": r["split"],
                    "z": zi,
                    "count": count,
                    "frac": frac,
                })
                if frac < min_z_frac and frac > 0 and frac < 0.01:
                    gate_errors.append((dataset_id, "z_bucket_lt_1pct", f"z={zi}"))

    if z_thresholds:
        pd.DataFrame(z_thresholds).to_csv(outdir / "Z_THRESHOLDS.csv", index=False)
    if z_distribution:
        pd.DataFrame(z_distribution).to_csv(outdir / "Z_DISTRIBUTION.csv", index=False)
    return z_thresholds, z_distribution, gate_errors


# --- Phase 3: P(z'|z) ---

def compute_p_z_given_z(manifest_df: pd.DataFrame, run_root: Path, z_thresholds: list, outdir: Path,
                       laplace_alpha: float = 1.0):
    """Time-ordered z sequence from train/val/fit; Laplace smoothing; row-stochastic 3x3."""
    run_root = Path(run_root)
    thresh = {(r["dataset_id"]): (r["q_lo"], r["q_hi"]) for r in z_thresholds}
    rows = []
    gate_errors = []

    for key, grp in manifest_df.groupby(["coin", "variant", "fold"]):
        coin, variant, fold = key
        dataset_id = f"{coin}_{variant}_fold{fold}"
        q_lo, q_hi = thresh.get(dataset_id, (-0.01, 0.01))
        z_seq = []
        for _, r in grp.iterrows():
            if r["split"] == "test":
                continue
            path = run_root / r["path"]
            kind = r["kind"]
            df, _ = load_and_canonicalize(path, run_root, kind, allow_no_pred_s7=True)
            if df is None or len(df) == 0:
                continue
            pred = df["pred"].values
            z = np.zeros(len(pred), dtype=int)
            z[pred >= q_hi] = 1
            z[pred <= q_lo] = -1
            z_seq.extend(z.tolist())
        if len(z_seq) < 2:
            gate_errors.append((dataset_id, "p_z_insufficient_rows", ""))
            continue
        z_seq = np.array(z_seq)
        trans = np.zeros((3, 3))
        for t in range(len(z_seq) - 1):
            i = int(z_seq[t]) + 1
            j = int(z_seq[t + 1]) + 1
            trans[i, j] += 1
        trans = trans + laplace_alpha
        row_sums = trans.sum(axis=1, keepdims=True)
        trans = trans / np.where(row_sums > 0, row_sums, 1)
        for i in range(3):
            s = trans[i, :].sum()
            if abs(s - 1) > 1e-9:
                gate_errors.append((dataset_id, "p_z_row_not_sum_1", str(s)))
            if np.any(np.isnan(trans[i, :])):
                gate_errors.append((dataset_id, "p_z_nan", ""))
            for j in range(3):
                rows.append({
                    "dataset_id": dataset_id,
                    "z": i - 1, "z_next": j - 1,
                    "P": float(trans[i, j]),
                })
    if rows:
        pd.DataFrame(rows).to_csv(outdir / "P_Z_GIVEN_Z.csv", index=False)
    return rows, gate_errors


# --- Z distribution shift (val vs test) ---

def compute_z_shift_metrics(manifest_df: pd.DataFrame, run_root: Path, z_thresholds: list, outdir: Path, eps: float = 1e-12):
    """
    Per dataset_id: p(z) from val, q(z) from test; write KL(val||test), KL(test||val), JS.
    Z_SHIFT_METRICS.csv: dataset_id, p_neg1, p_0, p_pos1, q_neg1, q_0, q_pos1, kl_val_test, kl_test_val, js_val_test.
    """
    run_root = Path(run_root)
    thresh = {r["dataset_id"]: (r["q_lo"], r["q_hi"]) for r in z_thresholds}
    rows = []
    for key, grp in manifest_df.groupby(["coin", "variant", "fold"]):
        coin, variant, fold = key
        dataset_id = f"{coin}_{variant}_fold{fold}"
        q_lo, q_hi = thresh.get(dataset_id, (-0.01, 0.01))
        p_vals = None
        q_vals = None
        for _, r in grp.iterrows():
            path = run_root / r["path"]
            if not path.exists():
                continue
            df, _ = load_and_canonicalize(path, run_root, r["kind"], allow_no_pred_s7=True)
            if df is None or len(df) == 0:
                continue
            pred = df["pred"].values
            z = np.zeros(len(pred), dtype=int)
            z[pred >= q_hi] = 1
            z[pred <= q_lo] = -1
            counts = [int((z == -1).sum()), int((z == 0).sum()), int((z == 1).sum())]
            probs = np.array(counts, dtype=float) / max(1, len(z))
            if r["split"] == "val":
                p_vals = probs
            elif r["split"] == "test":
                q_vals = probs
        if p_vals is None or q_vals is None:
            continue
        p = np.clip(p_vals + eps, 0, 1)
        q = np.clip(q_vals + eps, 0, 1)
        p = p / p.sum()
        q = q / q.sum()
        kl_val_test = float(np.sum(p * (np.log(p) - np.log(q))))
        kl_test_val = float(np.sum(q * (np.log(q) - np.log(p))))
        m = 0.5 * (p + q)
        js_val_test = float(0.5 * (np.sum(p * (np.log(p) - np.log(m))) + np.sum(q * (np.log(q) - np.log(m)))))
        rows.append({
            "dataset_id": dataset_id,
            "p_neg1": p[0], "p_0": p[1], "p_pos1": p[2],
            "q_neg1": q[0], "q_0": q[1], "q_pos1": q[2],
            "kl_val_test": kl_val_test,
            "kl_test_val": kl_test_val,
            "js_val_test": js_val_test,
        })
    if rows:
        pd.DataFrame(rows).to_csv(outdir / "Z_SHIFT_METRICS.csv", index=False)
    return rows


# --- Phase 4: Reward model ---

def _reward_fit_splits(reward_fit_mode: str) -> tuple:
    """Return (set of split names for reward aggregation, use_test70_truncation)."""
    if reward_fit_mode == "auto":
        return {"train", "val"}, False
    if reward_fit_mode == "train":
        return {"train"}, False
    if reward_fit_mode == "val":
        return {"val"}, False
    if reward_fit_mode == "val_and_test":
        return {"val", "test"}, False
    if reward_fit_mode == "test70":
        return {"test"}, True  # caller truncates test to first 70%
    return {"train", "val"}, False


def compute_reward_sanity_rows(df, split_name: str, fee_bps: float, q_lo: float, q_hi: float, dataset_id: str, kind: str = "step1") -> dict:
    """
    Compute one row of REWARD_SANITY for a given split dataframe.
    Uses pred to assign z via q_lo, q_hi; y for mean_y; r_long = y - fee_bps, r_short = -y - fee_bps.
    Returns dict with: dataset_id, kind, split, fee_bps, n, mean_y, std_y, p01_y, p99_y, zero_frac_y, nan_frac_y,
    mean_r_long, mean_r_short, n_z_neg1, n_z_0, n_z_pos1, mean_y_z_neg1, mean_y_z_0, mean_y_z_pos1.
    """
    if df is None or len(df) == 0:
        return None
    pred = df["pred"].values
    y = np.asarray(df["y"].values, dtype=float)
    z = np.zeros(len(pred), dtype=int)
    z[pred >= q_hi] = 1
    z[pred <= q_lo] = -1
    r_long = y - fee_bps
    r_short = -y - fee_bps
    n = len(y)
    mean_y = float(np.nanmean(y))
    std_y = float(np.nanstd(y)) if n > 1 else 0.0
    p01_y = float(np.nanpercentile(y, 1)) if n else 0.0
    p99_y = float(np.nanpercentile(y, 99)) if n else 0.0
    zero_frac_y = float(np.sum(np.abs(y) < 1e-9) / n) if n else 0.0
    nan_frac_y = float(np.sum(np.isnan(y)) / n) if n else 0.0
    mean_r_long = float(np.nanmean(r_long))
    mean_r_short = float(np.nanmean(r_short))
    n_z_neg1 = int((z == -1).sum())
    n_z_0 = int((z == 0).sum())
    n_z_pos1 = int((z == 1).sum())
    mean_y_z_neg1 = float(np.nanmean(y[z == -1])) if n_z_neg1 else np.nan
    mean_y_z_0 = float(np.nanmean(y[z == 0])) if n_z_0 else np.nan
    mean_y_z_pos1 = float(np.nanmean(y[z == 1])) if n_z_pos1 else np.nan
    return {
        "dataset_id": dataset_id,
        "kind": kind,
        "split": split_name,
        "fee_bps": fee_bps,
        "n": n,
        "mean_y": mean_y,
        "std_y": std_y,
        "p01_y": p01_y,
        "p99_y": p99_y,
        "zero_frac_y": zero_frac_y,
        "nan_frac_y": nan_frac_y,
        "mean_r_long": mean_r_long,
        "mean_r_short": mean_r_short,
        "n_z_neg1": n_z_neg1,
        "n_z_0": n_z_0,
        "n_z_pos1": n_z_pos1,
        "mean_y_z_neg1": mean_y_z_neg1,
        "mean_y_z_0": mean_y_z_0,
        "mean_y_z_pos1": mean_y_z_pos1,
    }


def reward_model_step1(manifest_df: pd.DataFrame, run_root: Path, z_thresholds: list, outdir: Path,
                       fee_bps: float, nmin_bucket: int, reward_fit_mode: str = "auto"):
    """
    Counterfactual: r_long = y - fee_bps, r_short = -y - fee_bps.
    Aggregate by z from splits per reward_fit_mode (auto|train|val|val_and_test|test70); store samples for rollouts.
    """
    run_root = Path(run_root)
    thresh = {r["dataset_id"]: (r["q_lo"], r["q_hi"]) for r in z_thresholds}
    reward_splits, use_test70_truncation = _reward_fit_splits(reward_fit_mode)
    stats_rows = []
    sanity_rows = []
    gate_errors = []

    for key, grp in manifest_df.groupby(["coin", "variant", "fold"]):
        coin, variant, fold = key
        dataset_id = f"{coin}_{variant}_fold{fold}"
        q_lo, q_hi = thresh.get(dataset_id, (-0.01, 0.01))
        r_long_by_z = defaultdict(list)
        r_short_by_z = defaultdict(list)
        for _, r in grp.iterrows():
            if r["split"] not in reward_splits:
                continue
            path = run_root / r["path"]
            if r["kind"] != "step1" or not path.exists():
                continue
            df, _ = load_and_canonicalize(path, run_root, "step1", False)
            if df is None or len(df) == 0:
                continue
            if use_test70_truncation and r["split"] == "test":
                n70 = max(1, int(0.7 * len(df)))
                df = df.iloc[:n70]
                if len(df) == 0:
                    continue
            pred = df["pred"].values
            y = df["y"].values
            z = np.zeros(len(pred), dtype=int)
            z[pred >= q_hi] = 1
            z[pred <= q_lo] = -1
            r_long = y - fee_bps
            r_short = -y - fee_bps
            for zi in (-1, 0, 1):
                mask = z == zi
                r_long_by_z[zi].extend(r_long[mask].tolist())
                r_short_by_z[zi].extend(r_short[mask].tolist())

        for zi in (-1, 0, 1):
            long_vals = r_long_by_z[zi]
            short_vals = r_short_by_z[zi]
            n_long = len(long_vals)
            n_short = len(short_vals)
            mean_long = float(np.mean(long_vals)) if n_long else 0.0
            std_long = float(np.std(long_vals)) if n_long > 1 else 0.0
            mean_short = float(np.mean(short_vals)) if n_short else 0.0
            std_short = float(np.std(short_vals)) if n_short > 1 else 0.0
            zero_frac_long = float(np.sum(np.abs(np.array(long_vals)) < 1e-9) / n_long) if n_long else 0.0
            zero_frac_short = float(np.sum(np.abs(np.array(short_vals)) < 1e-9) / n_short) if n_short else 0.0
            nan_frac_long = float(np.sum(np.isnan(long_vals)) / n_long) if n_long else 0.0
            nan_frac_short = float(np.sum(np.isnan(short_vals)) / n_short) if n_short else 0.0

            stats_rows.append({
                "dataset_id": dataset_id,
                "z": zi,
                "action": "long",
                "n": n_long,
                "mean_bps": mean_long,
                "std_bps": std_long,
                "zero_frac": zero_frac_long,
                "nan_frac": nan_frac_long,
            })
            stats_rows.append({
                "dataset_id": dataset_id,
                "z": zi,
                "action": "short",
                "n": n_short,
                "mean_bps": mean_short,
                "std_bps": std_short,
                "zero_frac": zero_frac_short,
                "nan_frac": nan_frac_short,
            })

        # Sanity per split: train, val, test (and test70/test30 if only test)
        has_test = any(r["split"] == "test" for _, r in grp.iterrows())
        for _, r in grp.iterrows():
            path = run_root / r["path"]
            if r["kind"] != "step1" or not path.exists():
                continue
            df_split, _ = load_and_canonicalize(path, run_root, "step1", False)
            if df_split is None or len(df_split) == 0:
                continue
            row = compute_reward_sanity_rows(df_split, r["split"], fee_bps, q_lo, q_hi, dataset_id, "step1")
            if row:
                sanity_rows.append(row)
        if has_test and not any(s["split"] == "test" for s in sanity_rows):
            pass  # will trigger WARN: test exists but not written
        # If only test exists, add test70_fit and test30_holdout
        if has_test:
            for _, r in grp.iterrows():
                if r["split"] != "test":
                    continue
                path = run_root / r["path"]
                if not path.exists():
                    continue
                df_test, _ = load_and_canonicalize(path, run_root, "step1", False)
                if df_test is None or len(df_test) < 2:
                    continue
                n70 = max(1, int(0.7 * len(df_test)))
                row70 = compute_reward_sanity_rows(df_test.iloc[:n70], "test70_fit", fee_bps, q_lo, q_hi, dataset_id, "step1")
                row30 = compute_reward_sanity_rows(df_test.iloc[n70:], "test30_holdout", fee_bps, q_lo, q_hi, dataset_id, "step1")
                if row70:
                    sanity_rows.append(row70)
                if row30:
                    sanity_rows.append(row30)
                break
        # When reward_fit_mode is val_and_test, add one sanity row for the combined fit split
        if reward_fit_mode == "val_and_test":
            val_df = test_df = None
            for _, r in grp.iterrows():
                path = run_root / r["path"]
                if r["kind"] != "step1" or not path.exists():
                    continue
                df, _ = load_and_canonicalize(path, run_root, "step1", False)
                if df is None or len(df) == 0:
                    continue
                if r["split"] == "val":
                    val_df = df
                elif r["split"] == "test":
                    test_df = df
            if val_df is not None and test_df is not None:
                combined = pd.concat([val_df, test_df], ignore_index=True)
                row_fit = compute_reward_sanity_rows(combined, "val_and_test", fee_bps, q_lo, q_hi, dataset_id, "step1")
                if row_fit:
                    sanity_rows.append(row_fit)

        # Gates
        all_long = [x for v in r_long_by_z.values() for x in v]
        all_short = [x for v in r_short_by_z.values() for x in v]
        any_std = any(
            (len(r_long_by_z[z]) > 0 and np.std(r_long_by_z[z]) > 1e-6) or
            (len(r_short_by_z[z]) > 0 and np.std(r_short_by_z[z]) > 1e-6)
            for z in (-1, 0, 1)
        )
        if not any_std:
            gate_errors.append((dataset_id, "reward_std_leq_1e6", ""))
        if all_long and np.sum(np.abs(np.array(all_long)) < 1e-9) / len(all_long) >= 0.98:
            gate_errors.append((dataset_id, "zero_frac_r_long_geq_98", ""))
        if all_short and np.sum(np.abs(np.array(all_short)) < 1e-9) / len(all_short) >= 0.98:
            gate_errors.append((dataset_id, "zero_frac_r_short_geq_98", ""))

    if stats_rows:
        pd.DataFrame(stats_rows).to_csv(outdir / "REWARD_STATS.csv", index=False)
    if sanity_rows:
        pd.DataFrame(sanity_rows).to_csv(outdir / "REWARD_SANITY.csv", index=False)

    # Build samples_by_dataset for rollouts (same reward_fit_mode splits)
    samples_by_dataset = {}
    for key, grp in manifest_df.groupby(["coin", "variant", "fold"]):
        coin, variant, fold = key
        dataset_id = f"{coin}_{variant}_fold{fold}"
        q_lo, q_hi = thresh.get(dataset_id, (-0.01, 0.01))
        sl = defaultdict(list)
        ss = defaultdict(list)
        for _, r in grp.iterrows():
            if r["split"] not in reward_splits:
                continue
            path = run_root / r["path"]
            if r["kind"] != "step1" or not path.exists():
                continue
            df, _ = load_and_canonicalize(path, run_root, "step1", False)
            if df is None or len(df) == 0:
                continue
            if use_test70_truncation and r["split"] == "test":
                n70 = max(1, int(0.7 * len(df)))
                df = df.iloc[:n70]
                if len(df) == 0:
                    continue
            pred = df["pred"].values
            y = df["y"].values
            z = np.zeros(len(pred), dtype=int)
            z[pred >= q_hi] = 1
            z[pred <= q_lo] = -1
            r_long = y - fee_bps
            r_short = -y - fee_bps
            for zi in (-1, 0, 1):
                mask = z == zi
                sl[zi].extend(r_long[mask].tolist())
                ss[zi].extend(r_short[mask].tolist())
        samples_by_dataset[dataset_id] = {"samples_long": dict(sl), "samples_short": dict(ss)}
    return stats_rows, sanity_rows, gate_errors, samples_by_dataset


def reward_model_s7(manifest_df: pd.DataFrame, run_root: Path, z_thresholds: list, outdir: Path, nmin_bucket: int,
                    reward_fit_mode: str = "auto"):
    """Empirical: mu_long[z] = mean(net_bps) of trades with side=+1 and z; mu_short[z] similarly. Backoff if sparse."""
    run_root = Path(run_root)
    thresh = {r["dataset_id"]: (r["q_lo"], r["q_hi"]) for r in z_thresholds}
    reward_splits, use_test70_truncation = _reward_fit_splits(reward_fit_mode)
    stats_rows = []
    sanity_rows = []
    gate_errors = []

    for key, grp in manifest_df.groupby(["coin", "variant", "fold"]):
        coin, variant, fold = key
        dataset_id = f"{coin}_{variant}_fold{fold}"
        q_lo, q_hi = thresh.get(dataset_id, (-0.01, 0.01))
        agg = defaultdict(list)  # (z, side) -> net_bps list
        for _, r in grp.iterrows():
            if r["split"] not in reward_splits:
                continue
            path = run_root / r["path"]
            if r["kind"] != "s7" or not path.exists():
                continue
            df, _ = load_and_canonicalize(path, run_root, "s7", True)
            if df is None or len(df) == 0:
                continue
            if use_test70_truncation and r["split"] == "test":
                n70 = max(1, int(0.7 * len(df)))
                df = df.iloc[:n70]
                if len(df) == 0:
                    continue
            pred = df["pred"].values
            z = np.zeros(len(pred), dtype=int)
            z[pred >= q_hi] = 1
            z[pred <= q_lo] = -1
            net = df["net_bps"].values
            side = df["side"].values
            for zi in (-1, 0, 1):
                for s in (-1, 1):
                    mask = (z == zi) & (side == s)
                    agg[(zi, s)].extend(net[mask].tolist())

        for zi in (-1, 0, 1):
            long_vals = agg[(zi, 1)]
            short_vals = agg[(zi, -1)]
            n_long = len(long_vals)
            n_short = len(short_vals)
            if n_long < nmin_bucket:
                long_vals = [x for (z, s) in [(zi, 1)] for x in agg[(z, s)]]
                all_long_side = [x for (z, s) in agg if s == 1 for x in agg[(z, s)]]
                if len(all_long_side) >= nmin_bucket:
                    long_vals = all_long_side
                    n_long = len(long_vals)
                else:
                    long_vals = []
                    n_long = 0
            if n_short < nmin_bucket:
                all_short_side = [x for (z, s) in agg if s == -1 for x in agg[(z, s)]]
                if len(all_short_side) >= nmin_bucket:
                    short_vals = all_short_side
                    n_short = len(short_vals)
                else:
                    short_vals = []
                    n_short = 0

            mean_long = float(np.mean(long_vals)) if long_vals else 0.0
            std_long = float(np.std(long_vals)) if len(long_vals) > 1 else 0.0
            mean_short = float(np.mean(short_vals)) if short_vals else 0.0
            std_short = float(np.std(short_vals)) if len(short_vals) > 1 else 0.0
            zero_frac_long = float(np.sum(np.abs(np.array(long_vals)) < 1e-9) / n_long) if n_long else 0.0
            zero_frac_short = float(np.sum(np.abs(np.array(short_vals)) < 1e-9) / n_short) if n_short else 0.0

            stats_rows.append({
                "dataset_id": dataset_id,
                "z": zi,
                "action": "long",
                "n": n_long,
                "mean_bps": mean_long,
                "std_bps": std_long,
                "zero_frac": zero_frac_long,
                "nan_frac": 0.0,
            })
            stats_rows.append({
                "dataset_id": dataset_id,
                "z": zi,
                "action": "short",
                "n": n_short,
                "mean_bps": mean_short,
                "std_bps": std_short,
                "zero_frac": zero_frac_short,
                "nan_frac": 0.0,
            })

        all_net = [x for v in agg.values() for x in v]
        sanity_rows.append({
            "dataset_id": dataset_id,
            "mean_net_bps": float(np.mean(all_net)) if all_net else 0,
            "std_net_bps": float(np.std(all_net)) if len(all_net) > 1 else 0,
            "zero_frac": float(np.sum(np.abs(np.array(all_net)) < 1e-9) / len(all_net)) if all_net else 0,
        })

        any_std = any(
            (len(agg[(z, 1)]) >= nmin_bucket and np.std(agg[(z, 1)]) > 1e-6) or
            (len(agg[(z, -1)]) >= nmin_bucket and np.std(agg[(z, -1)]) > 1e-6)
            for z in (-1, 0, 1)
        )
        if not any_std and all_net:
            gate_errors.append((dataset_id, "reward_std_leq_1e6_s7", ""))

    if stats_rows:
        pd.DataFrame(stats_rows).to_csv(outdir / "REWARD_STATS.csv", index=False)
    if sanity_rows:
        pd.DataFrame(sanity_rows).to_csv(outdir / "REWARD_SANITY.csv", index=False)
    return stats_rows, sanity_rows, gate_errors, {}
