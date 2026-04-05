import pandas as pd
from scipy import stats

# Load merged dataset
DATA_FILE = "merged.csv"

control_ids = [
    "d36d233490f119be6566b30183c657b0",
    "375a503eb1cb9b10c5a275e486c8b315",
    "4c8cc9bb1104707156e4d47326e337af",
    "99f2915cc9137479e31d1b7b229360c9",
    "4fb6bfbc1eaefeb98ebe5bf6769d08b5",
    "77d97fcf94d49505f8e19e928bb605fa",
]

ai_ids = [
    "2dcfb9153e74877f57a78d931018e07a",
    "9660ff4148b06bcb0e260dfb0f395c9f",
    "c078fffff1f376aa302cb384e2d35591",
    "41f9ff76349cde99684c7d97ac44e99b",
    "49ec194a70c7b9008a5a8b98f498c92c",
]

METRICS = [
    "accuracy",
    "avg_time_ms",
    "confidence_rating",
    "mental_effort"
]

# ===================================================
# LOAD DATA
# ===================================================

print("\nLoading data...")
df = pd.read_csv(DATA_FILE)

print(f"Loaded {len(df)} rows")

# ===================================================
# VALIDATE PARTICIPANT GROUPS
# ===================================================

print("\nChecking for overlapping participant IDs...")
overlap = set(control_ids) & set(ai_ids)

if overlap:
    print("WARNING: These participants appear in BOTH groups:")
    for oid in overlap:
        print("  ", oid)
    print("They will be assigned to the AI group due to overwrite.\n")
else:
    print("No overlap detected.\n")

# ===================================================
# ASSIGN CONDITIONS BASED ON WORKER ID
# ===================================================

df["condition"] = None

df.loc[df["session_id"].isin(control_ids), "condition"] = "baseline"
df.loc[df["session_id"].isin(ai_ids), "condition"] = "ai"

# Remove rows without a valid participant group
before = len(df)
df = df.dropna(subset=["condition"])
after = len(df)

print(f"Removed {before - after} rows with unknown participant IDs")
print(f"Remaining rows: {after}")

# ===================================================
# COMPUTE PER-SESSION METRICS
# ===================================================

print("\nComputing per-session metrics...")

session_metrics = (
    df.groupby(["session_id", "condition"])
    .agg({
        "is_correct": "mean",
        "duration_ms": "mean",
        "confidence_rating": "mean",
        "mental_effort": "mean"
    })
    .reset_index()
)

session_metrics.rename(columns={
    "is_correct": "accuracy",
    "duration_ms": "avg_time_ms"
}, inplace=True)

print(session_metrics.head())

# ===================================================
# SPLIT BY CONDITION
# ===================================================

baseline = session_metrics[session_metrics.condition == "baseline"]
ai = session_metrics[session_metrics.condition == "ai"]

print(f"\nSessions in baseline: {len(baseline)}")
print(f"Sessions in AI: {len(ai)}")

# ===================================================
# SUMMARY STATISTICS
# ===================================================

print("\n==============================")
print("SUMMARY STATISTICS")
print("==============================")

for metric in METRICS:
    print(f"\n{metric}")
    print("  baseline mean:", baseline[metric].mean())
    print("  ai mean:      ", ai[metric].mean())
print("\n==============================")
print("INDEPENDENT T-TESTS")
print("==============================")

def cohens_d_independent(x, y):
    nx, ny = len(x), len(y)
    pooled_std = (((nx-1)*x.std(ddof=1)**2 + (ny-1)*y.std(ddof=1)**2) / (nx+ny-2))**0.5
    return (x.mean() - y.mean()) / pooled_std

def independent_test(metric):
    b = baseline[metric].dropna()
    a = ai[metric].dropna()

    if len(b) < 2 or len(a) < 2:
        print(f"{metric}: Not enough data")
        return

    t, p = stats.ttest_ind(b, a, equal_var=False)
    d = cohens_d_independent(b, a)

    print(f"\n{metric}")
    print(f"  baseline mean = {b.mean():.4f}")
    print(f"  ai mean       = {a.mean():.4f}")
    print(f"  t = {t:.3f}")
    print(f"  p = {p:.4f}")
    print(f"  Cohen's d = {d:.3f}")

for metric in METRICS:
    independent_test(metric)

print("\nAnalysis complete.")


# ===================================================
# IMAGE-LEVEL ERROR ANALYSIS
# ===================================================

print("\n==============================")
print("MOST MISCLASSIFIED IMAGES")
print("==============================")

# Ensure your CSV has an image column (adjust name if needed)
IMAGE_COL = "image_name"   # change if your column is called something else

if IMAGE_COL not in df.columns:
    print(f"Column '{IMAGE_COL}' not found. Available columns:", df.columns.tolist())
else:
    # Compute per-image accuracy per condition
    image_metrics = (
        df.groupby([IMAGE_COL, "condition"])
        .agg({
            "is_correct": "mean",
            "session_id": "count"
        })
        .rename(columns={
            "is_correct": "accuracy",
            "session_id": "num_trials"
        })
        .reset_index()
    )

    # Compute error rate
    image_metrics["error_rate"] = 1 - image_metrics["accuracy"]

    # Split by condition
    worst_baseline = (
        image_metrics[image_metrics.condition == "baseline"]
        .sort_values(by="error_rate", ascending=False)
    )

    worst_ai = (
        image_metrics[image_metrics.condition == "ai"]
        .sort_values(by="error_rate", ascending=False)
    )

    TOP_K = 10

    print("\n--- Baseline: Most misclassified images ---")
    print(worst_baseline.head(TOP_K)[[IMAGE_COL, "error_rate", "num_trials"]])

    print("\n--- AI: Most misclassified images ---")
    print(worst_ai.head(TOP_K)[[IMAGE_COL, "error_rate", "num_trials"]])