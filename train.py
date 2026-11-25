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
CHUNK_SIZE = 100_000


def detect_label_column(columns: list[str]) -> str:
    for candidate in ("label", "Label"):
        if candidate in columns:
            return candidate
    non_source = [c for c in columns if c != "_source_file"]
    if not non_source:
        raise ValueError("Could not detect a label column in the CSV files.")
    return non_source[-1]


def collect_capped_rows(csv_files: list[str], max_total: int = TARGET_SAMPLE_ROWS) -> pd.DataFrame:
    collected = []
    total_kept = 0
    per_file_cap = max(25_000, max_total // max(1, len(csv_files)))

    for file_path in csv_files:
        if total_kept >= max_total:
            break

        stem = Path(file_path).stem
        file_kept = 0
        reader = pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False)

        for chunk in reader:
            remaining_global = max_total - total_kept
            remaining_file = per_file_cap - file_kept
            take_n = min(len(chunk), remaining_global, remaining_file)
            if take_n <= 0:
                break

            sampled_chunk = chunk.sample(n=take_n, random_state=42)
            sampled_chunk["_source_file"] = stem
            collected.append(sampled_chunk)

            total_kept += take_n
            file_kept += take_n

            if total_kept >= max_total or file_kept >= per_file_cap:
                break

    if not collected:
        raise ValueError("No rows could be read from raw_csv/.")

    return pd.concat(collected, ignore_index=True)


def main() -> None:
    csv_files = sorted(glob.glob(str(RAW_DIR / "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{RAW_DIR}'. Place CSV files inside raw_csv/ first."
        )

    df = collect_capped_rows(csv_files, max_total=TARGET_SAMPLE_ROWS)
    label_col = detect_label_column(list(df.columns))

    raw_labels = df[label_col].astype(str).str.strip()
    binary_target = np.where(raw_labels.str.lower() ==
                             "benign", "Benign", "Attack")
    attack_detail = raw_labels

    excluded_cols = {label_col, "_source_file"}
    feature_cols = [
        col for col in df.columns
        if col not in excluded_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns found in the dataset.")

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    medians = X.median(numeric_only=True).fillna(0.0)
    X = X.fillna(medians)
    y = binary_target

    stratify_y = y if pd.Series(y).value_counts().min() >= 2 else None
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
    macro_f1 = f1_score(y_test, preds, average="macro")

    print(
        f"Collected rows: {len(df):,} | Train: {len(X_train):,} | Test: {len(X_test):,} | Features: {len(feature_cols)}")
    print("Class counts in sampled data:",
          pd.Series(y).value_counts().to_dict())
    print(f"Macro F1: {macro_f1:.4f}\n")
    print(classification_report(y_test, preds, digits=4))

    model_bundle = {
        "model": clf,
        "feature_cols": feature_cols,
        "medians": medians.to_dict(),
    }
    model_path = MODEL_DIR / "iomt_rf_model.joblib"
    joblib.dump(model_bundle, model_path)

    test_meta = pd.DataFrame(
        {
            "binary_target": np.asarray(y_test),
            "attack_detail": np.asarray(attack_detail)[X_test.index],
            "predicted_label": preds,
        },
        index=X_test.index,
    )
    demo_pool = pd.concat([X_test, test_meta], axis=1)

    benign_rows = demo_pool[demo_pool["binary_target"] == "Benign"].head(250)
    attack_rows = demo_pool[demo_pool["binary_target"] == "Attack"].head(250)

    demo_df = (
        pd.concat([benign_rows, attack_rows], ignore_index=True)
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )
    demo_path = DATA_DIR / "demo_samples.csv"
    demo_df.to_csv(demo_path, index=False)

    print(f"Saved model bundle -> {model_path}")
    print(f"Saved demo samples -> {demo_path} ({len(demo_df)} rows)")


if __name__ == "__main__":
    main()
