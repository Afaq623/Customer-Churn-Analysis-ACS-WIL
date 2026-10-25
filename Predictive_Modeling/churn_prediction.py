"""
Customer Churn Prediction - Artificial Neural Network (Keras / TensorFlow)

Architecture (main model, as in the Stage 3 submission template):
    Input layer   : 12 features
    Hidden layer 1: 16 neurons, ReLU
    Hidden layer 2:  8 neurons, ReLU
    Output layer  :  1 neuron, Sigmoid (probability of churn)

Training: Adam optimiser, Binary Cross-Entropy loss, up to 50 epochs, batch size 32,
          20% of the training data held out for validation,
          early stopping on validation loss to optimise convergence.

Inputs : data/churn_train_processed.csv, data/churn_test_processed.csv
Outputs: churn_ann_model.h5, predictions.csv, evaluation_results.csv, model_comparison.csv,
         robustness_results.csv, confusion_matrix.png, accuracy_loss_graph.png, roc_curve.png
"""
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay)

# ------------------------------------------------------------------ paths
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
    raise FileNotFoundError(f"Cannot find '{name}'. Put it in: {HERE / 'data'}")


SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# ------------------------------------------------------------------ 1. data
print("=" * 60, "\nSTEP 1: Load train / test data\n" + "=" * 60)
train = pd.read_csv(find("churn_train_processed.csv"))
test = pd.read_csv(find("churn_test_processed.csv"))
FEATURES = [c for c in train.columns if c != "Churn"]
X_train_full, y_train_full = train[FEATURES].values.astype("float32"), train["Churn"].values
X_test, y_test = test[FEATURES].values.astype("float32"), test["Churn"].values
print(f"Train {X_train_full.shape}, Test {X_test.shape}, features = {len(FEATURES)}")
print(f"Churn rate  train {y_train_full.mean():.3f} | test {y_test.mean():.3f}")

# 20% of the training data is held out for validation (stratified so churn share is equal)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.20, stratify=y_train_full, random_state=SEED)


# ------------------------------------------------------------------ 2. models
def build_model(hidden=(16, 8), dropout=0.0, n_features=len(FEATURES)):
    m = keras.Sequential(name=f"ANN_{'_'.join(map(str, hidden))}")
    m.add(layers.Input(shape=(n_features,), name="Input_Layer"))
    for i, units in enumerate(hidden, 1):
        m.add(layers.Dense(units, activation="relu", name=f"Hidden_Layer_{i}"))
    if dropout:
        m.add(layers.Dropout(dropout, name="Dropout_Layer"))
    m.add(layers.Dense(1, activation="sigmoid", name="Output_Layer"))
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m


def train_model(hidden, dropout, seed):
    set_seed(seed)
    model = build_model(hidden, dropout)
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                         restore_best_weights=True)
    hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=50,
                     batch_size=32, callbacks=[stop], verbose=0)
    return model, hist


def metrics(y_true, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    return dict(Accuracy=accuracy_score(y_true, pred), Precision=precision_score(y_true, pred, zero_division=0),
                Recall=recall_score(y_true, pred), F1=f1_score(y_true, pred), AUC=roc_auc_score(y_true, prob))


def best_threshold(y_true, prob):
    """Pick the cut-off that maximises F1 on the VALIDATION set (never on the test set)."""
    grid = np.arange(0.10, 0.90, 0.01)
    return float(grid[int(np.argmax([f1_score(y_true, prob >= t) for t in grid]))])


print("\n" + "=" * 60, "\nSTEP 2: Train main ANN (16 -> 8 -> 1)\n" + "=" * 60)
model, hist = train_model((16, 8), 0.0, SEED)
model.summary()
epochs_run = len(hist.history["loss"])
best_epoch = int(np.argmin(hist.history["val_loss"])) + 1
print(f"Epochs run: {epochs_run} | best epoch (lowest validation loss): {best_epoch}")

# ------------------------------------------------------------------ 3. predict + evaluate
print("\n" + "=" * 60, "\nSTEP 3: Predict churn on the test set and evaluate\n" + "=" * 60)
p_val = model.predict(X_val, verbose=0).ravel()
p_test = model.predict(X_test, verbose=0).ravel()
thr = best_threshold(y_val, p_val)
print(f"Threshold chosen on validation data: {thr:.2f} (default 0.50)")

res = {"ANN 16-8 (threshold 0.50)": metrics(y_test, p_test, 0.5),
       f"ANN 16-8 (tuned threshold {thr:.2f})": metrics(y_test, p_test, thr)}

# Comparison model 1: the earlier 64-32 + dropout architecture, same training recipe
m2, h2 = train_model((64, 32), 0.2, SEED)
res["ANN 64-32 + Dropout 0.2 (threshold 0.50)"] = metrics(y_test, m2.predict(X_test, verbose=0).ravel(), 0.5)

# Comparison model 2: simple logistic regression baseline
lr = LogisticRegression(max_iter=1000).fit(X_train_full, y_train_full)
res["Logistic Regression baseline (threshold 0.50)"] = metrics(y_test, lr.predict_proba(X_test)[:, 1], 0.5)

comparison = pd.DataFrame(res).T.round(4)
print(comparison.to_string())
comparison.to_csv("model_comparison.csv")

# Official results = main model at the default 0.5 cut-off, plus tuned-threshold row
official = pd.DataFrame({"Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"],
                         "Value (threshold 0.50)": [f"{v*100:.1f}%" for v in res["ANN 16-8 (threshold 0.50)"].values()],
                         f"Value (tuned threshold {thr:.2f})": [f"{v*100:.1f}%" for v in res[f"ANN 16-8 (tuned threshold {thr:.2f})"].values()]})
official.to_csv("evaluation_results.csv", index=False)

# predictions file
pd.DataFrame({"actual_churn": y_test, "churn_probability": p_test.round(4),
              "predicted_default_0.50": (p_test >= 0.5).astype(int),
              f"predicted_tuned_{thr:.2f}": (p_test >= thr).astype(int)}).to_csv("predictions.csv", index=False)

# ------------------------------------------------------------------ 4. robustness (5 seeds)
print("\n" + "=" * 60, "\nSTEP 4: Robustness - repeat training with 5 different random seeds\n" + "=" * 60)
rows = []
for s in [1, 7, 21, 42, 99]:
    mm, _ = train_model((16, 8), 0.0, s)
    rows.append(dict(seed=s, **metrics(y_test, mm.predict(X_test, verbose=0).ravel(), 0.5)))
rob = pd.DataFrame(rows).round(4)
summary = pd.DataFrame({"mean": rob.drop(columns="seed").mean(), "std": rob.drop(columns="seed").std()}).round(4)
print(rob.to_string(index=False)); print(summary.T.to_string())
rob.to_csv("robustness_results.csv", index=False)

# ------------------------------------------------------------------ 5. charts + model file
model.save("churn_ann_model.h5")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].plot(hist.history["accuracy"], label="Training"); ax[0].plot(hist.history["val_accuracy"], label="Validation")
ax[0].axvline(best_epoch - 1, ls="--", c="grey", label=f"Best epoch ({best_epoch})")
ax[0].set(title="Model accuracy", xlabel="Epoch", ylabel="Accuracy"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(hist.history["loss"], label="Training"); ax[1].plot(hist.history["val_loss"], label="Validation")
ax[1].axvline(best_epoch - 1, ls="--", c="grey", label=f"Best epoch ({best_epoch})")
ax[1].set(title="Model loss", xlabel="Epoch", ylabel="Binary cross-entropy"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("accuracy_loss_graph.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(5, 4.5))
cm = confusion_matrix(y_test, (p_test >= 0.5).astype(int))
ConfusionMatrixDisplay(cm, display_labels=["No churn", "Churn"]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion matrix (test set, threshold 0.50)")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.close()

fpr, tpr, _ = roc_curve(y_test, p_test)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.plot(fpr, tpr, label=f"ANN 16-8 (AUC = {roc_auc_score(y_test, p_test):.3f})")
ax.plot([0, 1], [0, 1], "k--", label="Random guess")
ax.set(title="ROC curve (test set)", xlabel="False positive rate", ylabel="True positive rate")
ax.legend(loc="lower right"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("roc_curve.png", dpi=150); plt.close()

print("\nDone. Confusion matrix (rows = actual, columns = predicted):\n", cm)
