import glob
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

RAW_DIR = Path("raw_csv")
MODEL_DIR = Path("models")
DATA_DIR = Path("data")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SAMPLE_ROWS = 150_000
DEMO_PER_CLASS = 250
CHUNK_SIZE = 100_000


def detect_label_column(columns: list[str]) -> str:
    for candidate in ("label", "Label"):
        if candidate in columns:
            return candidate
    non_source = [c for c in columns if c != "_source_file"]
    if not non_source:
        raise ValueError("Could not detect a label column in the CSV files.")
    return non_source[-1]


def collect_balanced_demo_and_sample(
    csv_files: list[str],
    max_total: int = TARGET_SAMPLE_ROWS,
    demo_per_class: int = DEMO_PER_CLASS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sampled_parts = []
    benign_parts = []
    attack_parts = []

    total_kept = 0
    benign_kept = 0
    attack_kept = 0
    per_file_cap = max(25_000, max_total // max(1, len(csv_files)))

    for file_path in csv_files:
        if total_kept >= max_total and benign_kept >= demo_per_class and attack_kept >= demo_per_class:
            break

        stem = Path(file_path).stem
        file_kept = 0
        reader = pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False)

        for chunk in reader:
            if chunk.empty:
                continue

            label_col = detect_label_column(list(chunk.columns))
            raw_labels = chunk[label_col].astype(str).str.strip()
            is_benign = raw_labels.str.lower() == "benign"

            # Reserve up to demo_per_class rows per class for data/demo_samples.csv
            if benign_kept < demo_per_class:
                benign_chunk = chunk[is_benign]
                if not benign_chunk.empty:
                    take_b = min(len(benign_chunk), demo_per_class - benign_kept)
                    b_sample = benign_chunk.sample(n=take_b, random_state=42).copy()
                    b_sample["_source_file"] = stem
                    benign_parts.append(b_sample)
                    benign_kept += take_b

            if attack_kept < demo_per_class:
                attack_chunk = chunk[~is_benign]
                if not attack_chunk.empty:
                    take_a = min(len(attack_chunk), demo_per_class - attack_kept)
                    a_sample = attack_chunk.sample(n=take_a, random_state=42).copy()
                    a_sample["_source_file"] = stem
                    attack_parts.append(a_sample)
                    attack_kept += take_a

            # Also collect up to max_total rows for fast model training
            if total_kept < max_total and file_kept < per_file_cap:
                take_n = min(len(chunk), max_total - total_kept, per_file_cap - file_kept)
                if take_n > 0:
                    s_part = chunk.sample(n=take_n, random_state=42).copy()
                    s_part["_source_file"] = stem
                    sampled_parts.append(s_part)
                    total_kept += take_n
                    file_kept += take_n

            if total_kept >= max_total and benign_kept >= demo_per_class and attack_kept >= demo_per_class:
                break

    if not sampled_parts:
        raise ValueError("No rows could be read from raw_csv/.")

    # Merge reserved benign/attack rows into training pool if they weren't already included
    reserved_parts = benign_parts + attack_parts
    if reserved_parts:
        reserved_df = pd.concat(reserved_parts, ignore_index=True)
        train_pool_df = pd.concat(sampled_parts + [reserved_df], ignore_index=True)
        if len(train_pool_df) > max_total:
            train_pool_df = train_pool_df.sample(n=max_total, random_state=42).reset_index(drop=True)
        else:
            train_pool_df = train_pool_df.reset_index(drop=True)
    else:
        reserved_df = pd.DataFrame()
        train_pool_df = pd.concat(sampled_parts, ignore_index=True)

    return train_pool_df, reserved_df


def prepare_numeric_features(df: pd.DataFrame, feature_cols: list[str] | None = None, medians: dict | None = None):
    label_col = detect_label_column(list(df.columns))
    raw_labels = df[label_col].astype(str).str.strip()
    binary_target = pd.Series(
        np.where(raw_labels.str.lower() == "benign", "Benign", "Attack"),
        index=df.index,
    )
    attack_detail = pd.Series(raw_labels.values, index=df.index)

    if feature_cols is None:
        excluded_cols = {label_col, "_source_file"}
        feature_cols = [
            col for col in df.columns
            if col not in excluded_cols and pd.api.types.is_numeric_dtype(df[col])
        ]
    if not feature_cols:
        raise ValueError("No numeric feature columns found in the dataset.")

    X = pd.DataFrame(index=df.index)
    for col in feature_cols:
        if col in df.columns:
            X[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        else:
            X[col] = np.nan

    if medians is None:
        med_series = X.median(numeric_only=True).fillna(0.0)
        medians = med_series.to_dict()

    X = X.fillna(medians)
    return X, binary_target, attack_detail, feature_cols, medians


def main() -> None:
    csv_files = sorted(glob.glob(str(RAW_DIR / "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{RAW_DIR}'. Place CIC_IoMT_2024_WiFi_MQTT CSV files inside raw_csv/ first."
        )

    train_pool_df, reserved_demo_df = collect_balanced_demo_and_sample(
        csv_files,
        max_total=TARGET_SAMPLE_ROWS,
        demo_per_class=DEMO_PER_CLASS,
    )

    X, y, attack_detail, feature_cols, medians = prepare_numeric_features(train_pool_df)

    stratify_y = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=stratify_y,
        random_state=42,
    )

    clf = RandomForestClassifier(
        n_estimators=80,
        max_depth=16,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)

    print(f"Sampled rows: {len(X):,} | Train: {len(X_train):,} | Test: {len(X_test):,} | Features: {len(feature_cols)}")
    print("Class counts in sampled data:", y.value_counts().to_dict())
    print(f"Macro F1: {macro_f1:.4f}\n")
    print(classification_report(y_test, preds, digits=4, zero_division=0))

    model_bundle = {
        "model": clf,
        "feature_cols": feature_cols,
        "medians": medians,
    }
    model_path = MODEL_DIR / "iomt_rf_model.joblib"
    joblib.dump(model_bundle, model_path)

    # Build demo_samples.csv from reserved benign/attack rows (up to 250 each)
    if not reserved_demo_df.empty:
        demo_X, demo_y, demo_detail, _, _ = prepare_numeric_features(
            reserved_demo_df,
            feature_cols=feature_cols,
            medians=medians,
        )
        demo_preds = clf.predict(demo_X)
        demo_pool = demo_X.copy()
        demo_pool["binary_target"] = demo_y.values
        demo_pool["attack_detail"] = demo_detail.values
        demo_pool["predicted_label"] = demo_preds

        benign_rows = demo_pool[demo_pool["binary_target"] == "Benign"].head(DEMO_PER_CLASS)
        attack_rows = demo_pool[demo_pool["binary_target"] == "Attack"].head(DEMO_PER_CLASS)
    else:
        test_meta = pd.DataFrame(
            {
                "binary_target": y_test.values,
                "attack_detail": attack_detail.loc[X_test.index].values,
                "predicted_label": preds,
            },
            index=X_test.index,
        )
        demo_pool = pd.concat([X_test, test_meta], axis=1)
        benign_rows = demo_pool[demo_pool["binary_target"] == "Benign"].head(DEMO_PER_CLASS)
        attack_rows = demo_pool[demo_pool["binary_target"] == "Attack"].head(DEMO_PER_CLASS)

    demo_df = (
        pd.concat([benign_rows, attack_rows], ignore_index=True)
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )
    demo_path = DATA_DIR / "demo_samples.csv"
    demo_df.to_csv(demo_path, index=False)

    print(f"Saved model bundle -> {model_path}")
    print(f"Saved demo samples -> {demo_path} ({len(demo_df)} rows: {demo_df['binary_target'].value_counts().to_dict()})")


if __name__ == "__main__":
    main()
