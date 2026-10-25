"""
Customer Segmentation (K-means) on tenure and monthly charges.
Inputs : data/churn_clustering_ready.csv (scaled), data/Dataset_ATS_v2.csv (raw, for real units)
Outputs: cluster_elbow.png, cluster_segments.png, cluster_churn_rates.png, cluster_profiles.csv
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import os
from pathlib import Path

# Your project folder. If it exists it is used; otherwise the script's own folder is used.
# (Set to None before sharing the code publicly, so your username is not exposed.)
PROJECT_DIR = r"C:\Users\neesh\OneDrive\Desktop\WilProgram\Final"

if PROJECT_DIR and Path(PROJECT_DIR).is_dir():
    HERE = Path(PROJECT_DIR)
else:
    HERE = Path(__file__).resolve().parent
os.chdir(HERE)
(HERE / "data").mkdir(exist_ok=True)

def find(name):
    """Look for a data file in ./data first, then next to this script."""
    for p in (HERE / "data" / name, HERE / name):
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot find '{name}'. Put it in a 'data' folder next to this script: {HERE / 'data'}")

SEED = 42
scaled = pd.read_csv(find("churn_clustering_ready.csv"))
raw = pd.read_csv(find("Dataset_ATS_v2.csv"))
assert len(scaled) == len(raw)            # rows are in the same order (verified)
X = scaled[["tenure", "MonthlyCharges"]].values

# 1. Elbow + silhouette for k = 2..10
ks = range(2, 11)
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_))
    print(f"k={k}  inertia={km.inertia_:9.1f}  silhouette={sil[-1]:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(list(ks), inertia, "o-")
ax[0].set(title="Elbow method", xlabel="Number of clusters (k)", ylabel="Inertia (WCSS)")
ax[1].plot(list(ks), sil, "o-", color="tab:orange")
ax[1].set(title="Silhouette score", xlabel="Number of clusters (k)", ylabel="Silhouette")
for a in ax:
    a.grid(alpha=.3)
plt.tight_layout()
plt.savefig("cluster_elbow.png", dpi=150)
plt.close()

# 2. Final model: k = 4 (elbow bends at 4 and silhouette peaks at 4)
K = 4
km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(X)
raw["cluster"] = km.labels_
raw["churn_flag"] = (raw["Churn"] == "Yes").astype(int)

# 3. Interpret clusters in real units (months, $) and give plain-English names
prof = raw.groupby("cluster").agg(
    customers=("churn_flag", "size"),
    avg_tenure_months=("tenure", "mean"),
    avg_monthly_charge=("MonthlyCharges", "mean"),
    churn_rate=("churn_flag", "mean"),
    senior_share=("SeniorCitizen", "mean"),
).round(3)
prof["pct_of_customers"] = (prof.customers / len(raw) * 100).round(1)

t_med, m_med = raw.tenure.median(), raw.MonthlyCharges.median()
def name(r):
    t = "Long-tenure" if r.avg_tenure_months >= t_med else "Short-tenure"
    m = "high-spend" if r.avg_monthly_charge >= m_med else "low-spend"
    return f"{t}, {m}"
prof["segment"] = prof.apply(name, axis=1)
prof = prof.sort_values("avg_tenure_months")
print(prof.to_string())
prof.to_csv("cluster_profiles.csv")

# Business-friendly segment names
friendly = {"Short-tenure, low-spend": "New Budget",
            "Short-tenure, high-spend": "New Premium (high risk)",
            "Long-tenure, low-spend": "Loyal Budget",
            "Long-tenure, high-spend": "Loyal Premium"}
prof["name"] = prof["segment"].map(friendly)
prof.to_csv("cluster_profiles.csv")
raw["segment"] = raw["cluster"].map(prof["name"])
raw.drop(columns="churn_flag").to_csv("data/customers_with_segments.csv", index=False)

# 4. Visualise: labelled scatter + churn rate per segment
colors = {"New Budget": "tab:blue", "New Premium (high risk)": "tab:red",
          "Loyal Budget": "tab:green", "Loyal Premium": "tab:purple"}
fig, ax = plt.subplots(figsize=(8, 5.5))
for n, g in raw.groupby("segment"):
    ax.scatter(g.tenure, g.MonthlyCharges, s=8, alpha=.45, c=colors[n], label=None)
cent = km.cluster_centers_
tc = cent[:, 0] * raw.tenure.std(ddof=0) + raw.tenure.mean()
mc = cent[:, 1] * raw.MonthlyCharges.std(ddof=0) + raw.MonthlyCharges.mean()
ax.scatter(tc, mc, marker="X", s=200, c="black", edgecolor="white")
ax.set(title="Customer segments (K-means, k=4)", xlabel="Tenure (months)", ylabel="Monthly charges ($)")
from matplotlib.lines import Line2D
handles = [Line2D([], [], marker="o", ls="", color=colors[n], alpha=.7, label=f"{n} (n={(raw.segment==n).sum():,})") for n in colors]
handles.append(Line2D([], [], marker="X", ls="", color="black", markersize=9, label="Cluster centres"))
ax.legend(handles=handles, fontsize=8, loc="upper left"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("cluster_segments.png", dpi=150); plt.close()

order = prof.sort_values("churn_rate", ascending=False)
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(order["name"], order["churn_rate"] * 100, color=[colors[n] for n in order["name"]])
ax.axhline(raw.churn_flag.mean() * 100, ls="--", c="grey", label=f"Overall churn {raw.churn_flag.mean()*100:.1f}%")
for b, v in zip(bars, order["churn_rate"] * 100):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center")
ax.set(title="Churn rate by customer segment", ylabel="Churn rate (%)"); ax.legend()
plt.xticks(rotation=12); plt.tight_layout(); plt.savefig("cluster_churn_rates.png", dpi=150); plt.close()
print(prof[["name","customers","pct_of_customers","avg_tenure_months","avg_monthly_charge","churn_rate"]].to_string())