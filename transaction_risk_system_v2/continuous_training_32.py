# continuous_training_32.py
# ============================================================
# MODULE C / 3.2 — CONTINUOUS TRAINING AGENT (UNIVERSAL)
# - detects new data in DB
# - checks data drift (simple PSI)
# - retrains models (risk_level + verification_complexity)
# - versions models and logs metrics
# ============================================================

import os
import json
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


# ============================================================
# 0) SETTINGS (CHANGE ON COMPETITIONS)
# ============================================================

DB_PATH = "db/app.db"
LABELED_TABLE = "transactions_labeled"

TARGET_RISK = "risk_level"
TARGET_COMPLEX = "verification_complexity"

MODEL_ROOT = "models"
VERSIONS_DIR = os.path.join(MODEL_ROOT, "versions")
LOG_PATH = os.path.join(MODEL_ROOT, "training_log.csv")
STATE_PATH = os.path.join(MODEL_ROOT, "training_state.json")

RANDOM_STATE = 42
TEST_SIZE = 0.2

# training speed
MAX_TRAIN_ROWS = 300_000          # None / 100k / 300k / 500k
MIN_NEW_ROWS_TO_TRAIN = 10_000    # retrain only if enough new rows

# drift settings
DRIFT_NUM_COLS = ["amount", "hour", "rule_score", "anomaly_score", "risk_score"]  # <-- adjust
PSI_THRESHOLD = 0.2  # 0.1 small, 0.2 medium, 0.3+ strong drift

DROP_COLS = [TARGET_RISK, TARGET_COMPLEX]  # add "tr_datetime" if мешает


# ============================================================
# 1) UTIL: STATE
# ============================================================

def load_state(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_rowid": 0, "last_train_time": None}

def save_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ============================================================
# 2) UTIL: PSI (Population Stability Index)
# ============================================================

def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    # drop NaN
    expected = expected.dropna()
    actual = actual.dropna()
    if expected.empty or actual.empty:
        return 0.0

    # quantile bins on expected
    try:
        cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        if len(cuts) < 3:
            return 0.0
    except Exception:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=cuts)
    act_counts, _ = np.histogram(actual, bins=cuts)

    exp_perc = exp_counts / max(exp_counts.sum(), 1)
    act_perc = act_counts / max(act_counts.sum(), 1)

    # avoid zeros
    eps = 1e-6
    exp_perc = np.clip(exp_perc, eps, 1)
    act_perc = np.clip(act_perc, eps, 1)

    return float(np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc)))


def compute_drift(df_ref: pd.DataFrame, df_new: pd.DataFrame, cols: list[str]) -> dict:
    out = {}
    for c in cols:
        if c in df_ref.columns and c in df_new.columns:
            out[c] = psi(df_ref[c], df_new[c])
    out["psi_mean"] = float(np.mean(list(out.values()))) if out else 0.0
    out["psi_max"] = float(np.max(list(out.values()))) if out else 0.0
    return out


# ============================================================
# 3) UTIL: METRICS
# ============================================================

def eval_multiclass(y_true, y_pred, y_proba=None):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc_ovr": np.nan
    }
    if y_proba is not None:
        try:
            out["roc_auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
        except Exception:
            pass
    return out


# ============================================================
# 4) TRAIN PIPELINES (like in 3.1, fixed sparse vs dense)
# ============================================================

def make_preprocessors(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=["int64","int32","float64","float32"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    preprocess_onehot = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False))
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ],
        remainder="drop"
    )

    preprocess_ordinal = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            ]), cat_cols),
        ],
        remainder="drop"
    )
    return preprocess_onehot, preprocess_ordinal, num_cols, cat_cols


MODELS = {
    "LogReg": ("onehot", LogisticRegression(max_iter=3000, n_jobs=-1, class_weight="balanced")),
    "RandomForest": ("ordinal", RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced_subsample"
    )),
    "HistGB": ("ordinal", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
}


def train_and_select(X_train, y_train, X_test, y_test, preprocess_onehot, preprocess_ordinal, target_name: str):
    rows = []
    best_name, best_pipe, best_f1 = None, None, -1

    for name, (kind, clf) in MODELS.items():
        preprocess = preprocess_onehot if kind == "onehot" else preprocess_ordinal
        pipe = Pipeline([("preprocess", preprocess), ("model", clf)])
        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)
        proba = None
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            try:
                proba = pipe.predict_proba(X_test)
            except Exception:
                pass

        m = eval_multiclass(y_test, pred, proba)
        m["model"] = name
        m["target"] = target_name
        rows.append(m)

        if m["f1_macro"] > best_f1:
            best_f1 = m["f1_macro"]
            best_name = name
            best_pipe = pipe

    res = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    return res, best_name, best_pipe


# ============================================================
# 5) MAIN: LOAD NEW DATA + DRIFT + TRAIN
# ============================================================

def main():
    os.makedirs(VERSIONS_DIR, exist_ok=True)
    os.makedirs(MODEL_ROOT, exist_ok=True)

    state = load_state(STATE_PATH)
    last_rowid = int(state.get("last_rowid", 0))

    con = sqlite3.connect(DB_PATH)

    # How many rows now?
    total_rows = pd.read_sql(f"SELECT COUNT(*) AS n FROM {LABELED_TABLE}", con)["n"].iloc[0]

    # Load NEW rows by rowid
    df_new = pd.read_sql(
        f"SELECT rowid AS _rowid_, * FROM {LABELED_TABLE} WHERE rowid > {last_rowid}",
        con
    )

    print(f"[INFO] total_rows={total_rows:,} | last_rowid={last_rowid:,} | new_rows={len(df_new):,}")

    if len(df_new) < MIN_NEW_ROWS_TO_TRAIN:
        print("[SKIP] Not enough new data to retrain. Exiting.")
        con.close()
        return

    # Load reference sample = previous data (small slice) for drift comparison
    # (take last 100k older rows)
    df_ref = pd.read_sql(
        f"""
        SELECT rowid AS _rowid_, * FROM {LABELED_TABLE}
        WHERE rowid <= {last_rowid}
        ORDER BY rowid DESC
        LIMIT 100000
        """,
        con
    )

    con.close()

    # Drift check
    drift = compute_drift(df_ref, df_new, DRIFT_NUM_COLS) if len(df_ref) > 0 else {"psi_mean": 0.0, "psi_max": 0.0}
    drift_flag = drift.get("psi_max", 0.0) >= PSI_THRESHOLD

    print("[DRIFT] psi_mean={:.4f} psi_max={:.4f} flag={}".format(
        drift.get("psi_mean", 0.0), drift.get("psi_max", 0.0), drift_flag
    ))

    # Training set = previous + new (or only recent slice)
    # Universal approach: retrain on recent window (fast + adapts)
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        f"""
        SELECT rowid AS _rowid_, * FROM {LABELED_TABLE}
        ORDER BY rowid DESC
        LIMIT {MAX_TRAIN_ROWS if MAX_TRAIN_ROWS is not None else 2000000}
        """,
        con
    )
    con.close()

    df = df.drop(columns=["_rowid_"], errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan)

    # Safety targets
    for t in [TARGET_RISK, TARGET_COMPLEX]:
        if t not in df.columns:
            raise ValueError(f"Missing target: {t}")

    # Build X,y
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].copy()
    y_risk = df[TARGET_RISK].astype(str)
    y_cx = df[TARGET_COMPLEX].astype(str)

    preprocess_onehot, preprocess_ordinal, num_cols, cat_cols = make_preprocessors(X)

    X_train, X_test, y_risk_train, y_risk_test, y_cx_train, y_cx_test = train_test_split(
        X, y_risk, y_cx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_risk
    )

    # Train both targets
    res_risk, best_risk_name, best_risk_pipe = train_and_select(
        X_train, y_risk_train, X_test, y_risk_test, preprocess_onehot, preprocess_ordinal, "risk_level"
    )
    res_cx, best_cx_name, best_cx_pipe = train_and_select(
        X_train, y_cx_train, X_test, y_cx_test, preprocess_onehot, preprocess_ordinal, "verification_complexity"
    )

    # Version tag
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = f"v_{ts}"

    risk_path = os.path.join(VERSIONS_DIR, f"{version}__best_model_risk__{best_risk_name}.joblib")
    cx_path = os.path.join(VERSIONS_DIR, f"{version}__best_model_complexity__{best_cx_name}.joblib")
    joblib.dump(best_risk_pipe, risk_path)
    joblib.dump(best_cx_pipe, cx_path)

    # Log row
    def pick_best(res_df: pd.DataFrame) -> dict:
        r = res_df.iloc[0].to_dict()
        return r

    row_risk = pick_best(res_risk)
    row_cx = pick_best(res_cx)

    log_row = {
        "timestamp": ts,
        "version": version,
        "trained_rows": len(df),
        "new_rows_detected": int(len(df_new)),
        "drift_psi_mean": drift.get("psi_mean", 0.0),
        "drift_psi_max": drift.get("psi_max", 0.0),
        "drift_flag": int(drift_flag),

        "risk_model": best_risk_name,
        "risk_accuracy": row_risk.get("accuracy"),
        "risk_recall_macro": row_risk.get("recall_macro"),
        "risk_f1_macro": row_risk.get("f1_macro"),
        "risk_roc_auc_ovr": row_risk.get("roc_auc_ovr"),

        "cx_model": best_cx_name,
        "cx_accuracy": row_cx.get("accuracy"),
        "cx_recall_macro": row_cx.get("recall_macro"),
        "cx_f1_macro": row_cx.get("f1_macro"),
        "cx_roc_auc_ovr": row_cx.get("roc_auc_ovr"),

        "risk_model_path": risk_path,
        "cx_model_path": cx_path,
    }

    log_df = pd.DataFrame([log_row])

    if os.path.exists(LOG_PATH):
        old = pd.read_csv(LOG_PATH)
        out = pd.concat([old, log_df], ignore_index=True)
    else:
        out = log_df

    out.to_csv(LOG_PATH, index=False)
    print("[SAVED] log ->", LOG_PATH)
    print("[SAVED] model risk ->", risk_path)
    print("[SAVED] model cx   ->", cx_path)

    # Update state: set last_rowid to current max rowid
    con = sqlite3.connect(DB_PATH)
    max_rowid = pd.read_sql(f"SELECT MAX(rowid) AS m FROM {LABELED_TABLE}", con)["m"].iloc[0]
    con.close()

    state["last_rowid"] = int(max_rowid)
    state["last_train_time"] = ts
    save_state(STATE_PATH, state)

    print("[STATE] updated ->", STATE_PATH)


if __name__ == "__main__":
    main()