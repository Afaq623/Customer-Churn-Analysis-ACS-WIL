"""
Churn driver analysis - which factors drive churn and retention?

Three complementary views:
  1. ANN permutation importance : how much does the trained model's AUC drop when a factor is shuffled?
  2. Logistic-regression odds ratios : direction and size of each factor's effect.
  3. Churn rate by group + chi-square tests on the raw data : plain-language evidence.

Inputs : churn_ann_model.h5, data/churn_test_processed.csv, data/churn_train_processed.csv,
         data/Dataset_ATS_v2.csv
Outputs: feature_importance.csv, churn_rate_by_factor.csv, statistical_tests.csv,
         driver_importance.png, churn_by_tenure_charges.png, churn_by_customer_profile.png,
         churn_contract_internet.png
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow import keras
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ------------------------------------------------------------------ paths
# Your project folder. If it exists it is used; otherwise the script's own folder is used.
# (Set to None before sharing the code publicly, so your username is not exposed.)
PROJECT_DIR = r"C:\Users\neesh\OneDrive\Desktop\WilProgram\Final"

if PROJECT_DIR and Path(PROJECT_DIR).is_dir():
    HERE = Path(PROJECT_DIR)
else:
    HERE = Path(__file__).resolve().parent
os.chdir(HERE)


def find(name):
    for p in (HERE / "data" / name, HERE / name):
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find '{name}'. Put it in: {HERE / 'data'}")


SEED = 42
rng = np.random.default_rng(SEED)

train = pd.read_csv(find("churn_train_processed.csv"))
test = pd.read_csv(find("churn_test_processed.csv"))
raw = pd.read_csv(find("Dataset_ATS_v2.csv"))
FEATURES = [c for c in train.columns if c != "Churn"]
X_test, y_test = test[FEATURES].copy(), test["Churn"].values
raw["churn"] = (raw["Churn"] == "Yes").astype(int)
overall = raw.churn.mean()

# ------------------------------------------------------------------ 1. permutation importance (ANN)
model = keras.models.load_model(find("churn_ann_model.h5"), compile=False)
base_auc = roc_auc_score(y_test, model.predict(X_test.values.astype("float32"), verbose=0).ravel())

GROUPS = {  # one-hot columns are shuffled together so a factor is judged as a whole
    "Tenure": ["tenure"],
    "Monthly charges": ["MonthlyCharges"],
    "Dependents": ["Dependents"],
    "Senior citizen": ["SeniorCitizen"],
    "Contract type": ["Contract_Month-to-month", "Contract_One year", "Contract_Two year"],
    "Internet service": ["InternetService_DSL", "InternetService_Fiber optic"],
    "Multiple lines": ["MultipleLines"],
    "Phone service": ["PhoneService"],
    "Gender": ["gender"],
}
REPEATS = 30
rows = []
for name, cols in GROUPS.items():
    drops = []
    for _ in range(REPEATS):
        Xp = X_test.copy()
        perm = rng.permutation(len(Xp))
        Xp[cols] = Xp[cols].values[perm]          # shuffle the factor's columns together
        auc = roc_auc_score(y_test, model.predict(Xp.values.astype("float32"), verbose=0).ravel())
        drops.append(base_auc - auc)
    rows.append((name, np.mean(drops), np.std(drops)))
imp = pd.DataFrame(rows, columns=["factor", "auc_drop", "auc_drop_std"]).sort_values("auc_drop", ascending=False)
print(f"Baseline test AUC: {base_auc:.4f}\n")
print(imp.round(4).to_string(index=False))

# ------------------------------------------------------------------ 2. logistic-regression odds ratios
lr = LogisticRegression(max_iter=1000).fit(train[FEATURES], train["Churn"])
odds = pd.Series(np.exp(lr.coef_[0]), index=FEATURES)
imp["odds_ratio_note"] = imp["factor"].map({
    "Tenure": f"x{odds['tenure']:.2f} per +1 SD (about +24.6 months)",
    "Monthly charges": f"x{odds['MonthlyCharges']:.2f} per +1 SD (about +$30)",
    "Dependents": f"x{odds['Dependents']:.2f} if customer has dependents",
    "Senior citizen": f"x{odds['SeniorCitizen']:.2f} if senior",
}).fillna("no meaningful effect")
imp.round(4).to_csv("feature_importance.csv", index=False)
print("\nOdds ratios (>1 raises churn risk, <1 lowers it):")
print(odds.round(3).to_string())

# ------------------------------------------------------------------ 3. churn rate by group + chi-square
raw["Tenure band"] = pd.cut(raw.tenure, [-1, 12, 24, 48, 72], labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"])
raw["Charges band"] = pd.cut(raw.MonthlyCharges, [0, 40, 70, 95, 200], labels=["< $40", "$40-70", "$70-95", "> $95"])
raw["Senior"] = raw.SeniorCitizen.map({0: "Not senior", 1: "Senior"})
FACTORS = ["Tenure band", "Charges band", "Senior", "Dependents", "Contract", "InternetService",
           "PhoneService", "MultipleLines", "gender"]

tables, tests = [], []
for f in FACTORS:
    g = raw.groupby(f, observed=True).churn.agg(customers="size", churn_rate="mean").reset_index()
    g.insert(0, "factor", f)
    g = g.rename(columns={f: "group"})
    tables.append(g)
    chi2, p, _, _ = chi2_contingency(pd.crosstab(raw[f], raw.churn))
    tests.append(dict(factor=f, chi2=round(chi2, 1), p_value=p,
                      significant_at_5pct="Yes" if p < 0.05 else "No"))
rates = pd.concat(tables, ignore_index=True)
rates["churn_rate"] = rates.churn_rate.round(4)
rates.to_csv("churn_rate_by_factor.csv", index=False)
tests = pd.DataFrame(tests)
tests["p_value"] = tests.p_value.map(lambda v: f"{v:.2e}")
tests.to_csv("statistical_tests.csv", index=False)
print("\nChurn rate by group:\n", rates.to_string(index=False))
print("\nChi-square tests (is the factor related to churn?):\n", tests.to_string(index=False))

# ------------------------------------------------------------------ 4. charts
RED, BLUE, GREY = "#d62728", "#1f77b4", "#9aa0a6"


def bar_panel(ax, factor, title, order=None):
    g = rates[rates.factor == factor].set_index("group")
    if order:
        g = g.loc[order]
    bars = ax.bar(g.index.astype(str), g.churn_rate * 100,
                  color=[RED if r > overall else BLUE for r in g.churn_rate])
    ax.axhline(overall * 100, ls="--", c="grey", lw=1)
    for b, v in zip(bars, g.churn_rate * 100):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_title(title); ax.set_ylabel("Churn rate (%)"); ax.set_ylim(0, max(g.churn_rate * 100) * 1.2)


# importance chart
fig, ax = plt.subplots(figsize=(8, 4.8))
d = imp.sort_values("auc_drop")
ax.barh(d.factor, d.auc_drop, xerr=d.auc_drop_std, color=[RED if v > 0.01 else GREY for v in d.auc_drop])
ax.set(title="What the ANN relies on (permutation importance)",
       xlabel="Drop in test AUC when the factor is shuffled (larger = more important)")
ax.grid(axis="x", alpha=.3)
plt.tight_layout(); plt.savefig("driver_importance.png", dpi=150); plt.close()

# tenure + charges
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
bar_panel(ax[0], "Tenure band", "Churn rate by tenure", ["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"])
bar_panel(ax[1], "Charges band", "Churn rate by monthly charges", ["< $40", "$40-70", "$70-95", "> $95"])
plt.suptitle(f"Dashed line = overall churn rate ({overall*100:.1f}%)", y=0.012, fontsize=9, va="bottom")
plt.tight_layout(rect=(0, 0.07, 1, 1)); plt.savefig("churn_by_tenure_charges.png", dpi=150); plt.close()

# customer profile
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
bar_panel(ax[0], "Senior", "Churn rate by senior status")
bar_panel(ax[1], "Dependents", "Churn rate by dependents")
plt.suptitle(f"Dashed line = overall churn rate ({overall*100:.1f}%)", y=0.012, fontsize=9, va="bottom")
plt.tight_layout(rect=(0, 0.07, 1, 1)); plt.savefig("churn_by_customer_profile.png", dpi=150); plt.close()

# contract + internet (shows no difference in this dataset)
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
bar_panel(ax[0], "Contract", "Churn rate by contract type")
bar_panel(ax[1], "InternetService", "Churn rate by internet service")
plt.suptitle("Almost no difference between groups in this dataset", y=0.012, fontsize=9, va="bottom")
plt.tight_layout(rect=(0, 0.07, 1, 1)); plt.savefig("churn_contract_internet.png", dpi=150); plt.close()

# tenure x monthly-charges heatmap (how the two strongest drivers combine)
combo = raw.pivot_table(index="Tenure band", columns="Charges band", values="churn", aggfunc=["mean", "size"], observed=True)
rate, cnt = combo["mean"].copy(), combo["size"].copy()
# plain text labels (category-typed labels break .join() in newer pandas versions)
rate.index, cnt.index = rate.index.astype(str), cnt.index.astype(str)
rate.columns, cnt.columns = rate.columns.astype(str), cnt.columns.astype(str)
out = pd.concat([rate.round(4).add_suffix(" rate"), cnt.add_suffix(" n")], axis=1)
out.to_csv("churn_rate_tenure_x_charges.csv")
fig, ax = plt.subplots(figsize=(7.4, 4.6))
im = ax.imshow(rate.values * 100, cmap="RdYlGn_r", vmin=0, vmax=60, aspect="auto")
ax.set_xticks(range(rate.shape[1])); ax.set_xticklabels(rate.columns)
ax.set_yticks(range(rate.shape[0])); ax.set_yticklabels(rate.index)
for i in range(rate.shape[0]):
    for j in range(rate.shape[1]):
        v = rate.values[i, j] * 100
        ax.text(j, i, f"{v:.1f}%\n(n={cnt.values[i, j]:,})", ha="center", va="center", fontsize=9,
                color="white" if (v >= 45 or v <= 8) else "black")
ax.set(title="Churn rate by tenure and monthly charges", xlabel="Monthly charges", ylabel="Tenure")
fig.colorbar(im, ax=ax, label="Churn rate (%)")
plt.tight_layout(); plt.savefig("churn_heatmap_tenure_charges.png", dpi=150); plt.close()
print("\nChurn rate, tenure band x charges band:\n", rate.round(3).to_string())

# dependents and seniors within tenure bands
dep = raw.pivot_table(index="Tenure band", columns="Dependents", values="churn", aggfunc="mean", observed=True).round(3)
dep.to_csv("churn_rate_tenure_x_dependents.csv")
print("\nChurn rate, tenure band x dependents:\n", dep.to_string())

print("\nCharts saved.")