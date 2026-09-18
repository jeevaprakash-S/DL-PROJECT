"""Leakage-safe CICIDS preprocessing.

Sequences never cross a source-file, destination-host, or temporal split
boundary.  The median imputer and scaler are fit on raw *training flows only*.
""" 
import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
import config

def find_column(columns, candidates):
    lookup = {str(c).strip().lower(): c for c in columns}
    return next((lookup[c.lower()] for c in candidates if c.lower() in lookup), None)

def read_csv(path):
    last = None
    for encoding in ("utf-8", "latin1"):
        try:
            df = pd.read_csv(path, encoding=encoding, low_memory=False)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except (UnicodeDecodeError, pd.errors.ParserError) as exc: last = exc
    raise RuntimeError(f"Could not read {path}: {last}")

def stage_for(label):
    value = str(label).strip().lower()
    for terms, stage in config.LABEL_TO_STAGE_RULES:
        if any(term in value for term in terms): return stage
    return None

def load_and_audit():
    paths = sorted(config.RAW_DATA_DIR.rglob("*.csv"))
    if not paths: raise FileNotFoundError(f"Put CICIDS CSV files under {config.RAW_DATA_DIR}")
    frames, audit = [], {"raw_labels": {}, "unmapped_labels": {}}
    for file_id, path in enumerate(paths):
        df = read_csv(path); label_col = find_column(df.columns, config.LABEL_COLUMN_CANDIDATES)
        if label_col is None: raise ValueError(f"No Label-like column in {path.name}")
        labels = df[label_col].astype(str).str.strip()
        for label, n in labels.value_counts().items(): audit["raw_labels"][label] = audit["raw_labels"].get(label, 0) + int(n)
        stages = labels.map(stage_for)
        for label, n in labels[stages.isna()].value_counts().items(): audit["unmapped_labels"][label] = audit["unmapped_labels"].get(label, 0) + int(n)
        # Unknown labels are not silently relabelled as a security stage.
        df = df.loc[stages.notna()].copy(); df["__stage__"] = stages[stages.notna()].values; df["__file__"] = path.name
        time_col = find_column(df.columns, config.TIMESTAMP_CANDIDATES)
        parsed = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True) if time_col else pd.Series(pd.NaT, index=df.index)
        df["__time__"] = parsed
        df["__row__"] = np.arange(len(df)); frames.append(df)
        print(f"{path.name}: retained {len(df):,}; unmapped {int(stages.isna().sum()):,}")
    return frames, audit

def numeric_features(frames):
    excluded = {"__stage__", "__file__", "__time__", "__row__", "Flow ID", "Source IP", "Destination IP", "Src IP", "Dst IP", "Timestamp", "Label"}
    candidates = None
    for df in frames:
        good = set()
        for col in df.columns:
            if col in excluded: continue
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().mean() >= config.MIN_NUMERIC_FRACTION: good.add(col)
        candidates = good if candidates is None else candidates & good
    if not candidates: raise ValueError("No shared numeric CIC flow features found.")
    return sorted(candidates)

def group_key(frame):
    requested = config.GROUP_BY_COLUMN

    # None means: keep one chronological stream per source CSV/day.
    if requested is None:
        return frame["__file__"].astype(str)

    group_col = (
        requested
        if requested in frame.columns
        else find_column(frame.columns, config.GROUP_COLUMN_CANDIDATES)
    )

    if group_col is None:
        print(f"[warn] {requested!r} missing; using source-file stream instead")
        return frame["__file__"].astype(str)

    return (
        frame["__file__"].astype(str)
        + "|"
        + frame[group_col].fillna("<missing>").astype(str)
    )
def time_split(frame):
    result = {"train": [], "val": [], "test": []}
    frame = frame.copy(); frame["__group__"] = group_key(frame)
    for _, group in frame.groupby("__group__", sort=False):
        group = group.sort_values(["__time__", "__row__"], kind="stable", na_position="last")
        n = len(group)
        if n < config.MIN_GROUP_ROWS: continue
        a, b = int(n * config.TRAIN_FRACTION), int(n * (config.TRAIN_FRACTION + config.VAL_FRACTION))
        # reserve enough raw flows for an independent window in each split
        if min(a, b-a, n-b) < config.WINDOW_SIZE + config.FORECAST_HORIZON: continue
        result["train"].append(group.iloc[:a]); result["val"].append(group.iloc[a:b]); result["test"].append(group.iloc[b:])
    return {name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame() for name, parts in result.items()}

def make_sequences(frame, features):
    x, y_next, y_stage = [], [], []
    for _, g in frame.groupby("__group__", sort=False):
        a = g[features].to_numpy(np.float32); labels = g["__stage__"].map(config.STAGE_TO_IDX).to_numpy(np.int64)
        last = len(g) - config.WINDOW_SIZE - config.FORECAST_HORIZON + 1
        for start in range(0, max(0, last), config.STRIDE):
            end, target = start + config.WINDOW_SIZE, start + config.WINDOW_SIZE + config.FORECAST_HORIZON - 1
            x.append(a[start:end]); y_next.append(a[target]); y_stage.append(labels[target])
    if not x: return np.empty((0, config.WINDOW_SIZE, len(features)), np.float32), np.empty((0, len(features)), np.float32), np.empty(0, np.int64)
    return np.asarray(x, np.float32), np.asarray(y_next, np.float32), np.asarray(y_stage, np.int64)

def run(inspect_only=False):
    config.ensure_dirs(); frames, audit = load_and_audit()
    audit_path = config.OUTPUT_DIR / "label_audit.json"
    if inspect_only:
        audit_path.write_text(json.dumps(audit, indent=2)); print(json.dumps(audit, indent=2)); return
    features = numeric_features(frames)
    raw_splits = time_split(pd.concat(frames, ignore_index=True))
    if raw_splits["train"].empty: raise RuntimeError("No groups were long enough to create train/val/test windows.")
    # Train-only statistics: this is intentionally before transforming val/test.
    train_num = raw_splits["train"][features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = train_num.median().fillna(0.0); keep = train_num.fillna(medians).var() > 1e-12
    features = [f for f in features if keep[f]]
    scaler = StandardScaler().fit(train_num[features].fillna(medians[features]))
    totals = {}
    for name, split in raw_splits.items():
        numeric = split[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(medians[features])
        scaled_features = pd.DataFrame(
            scaler.transform(numeric).astype(np.float32),
            columns=features,
            index=split.index,
        )
        split = pd.concat([split.drop(columns=features), scaled_features], axis=1)
        X, yn, ys = make_sequences(split, features)
        np.save(config.PROCESSED_DATA_DIR / f"X_{name}.npy", X); np.save(config.PROCESSED_DATA_DIR / f"y_next_{name}.npy", yn); np.save(config.PROCESSED_DATA_DIR / f"y_stage_{name}.npy", ys)
        totals[name] = {stage: int(n) for stage, n in zip(config.STAGES, np.bincount(ys, minlength=config.NUM_STAGES))}
        print(f"{name}: {len(X):,} sequences; {totals[name]}")
    audit.update({"mapped_stage_counts_by_split": totals, "features": features, "notes": "Unmapped labels are excluded and require an explicit mapping review."})
    audit_path.write_text(json.dumps(audit, indent=2)); (config.PROCESSED_DATA_DIR / "feature_cols.txt").write_text("\n".join(features)); joblib.dump({"scaler": scaler, "medians": medians, "features": features}, config.PROCESSED_DATA_DIR / "preprocessor.joblib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--inspect", action="store_true", help="write raw/unmapped label audit only")
    run(parser.parse_args().inspect)
