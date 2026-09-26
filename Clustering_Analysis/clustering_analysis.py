"""Stage 2: Customer Clustering Analysis

This script:
1. Loads the prepared customer dataset.
2. Uses tenure and MonthlyCharges for clustering.
3. Splits the data into training and testing sets.
4. Applies scaling only when the selected columns are not already standardised.
5. Uses the Elbow Method to select a working number of clusters.
6. Trains a K-Means model.
7. Creates cluster summaries and visualisations.

Required packages:
pandas, numpy, matplotlib, scikit-learn, joblib
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


INPUT_FILE = Path("churn_clustering_ready (2).csv")
OUTPUT_DIR = Path("Stage2_Clustering_Output")
RANDOM_STATE = 42
SELECTED_K = 4
FEATURES = ["tenure", "MonthlyCharges"]


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load data
    df = pd.read_csv(INPUT_FILE)
    missing = [column for column in FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep rows with valid clustering values
    df = df.dropna(subset=FEATURES).copy()
    X = df[FEATURES].astype(float)

    # Train/test split
    X_train, X_test = train_test_split(
        X, test_size=0.20, random_state=RANDOM_STATE
    )

    # Detect whether the features already appear standardised.
    # If not, apply StandardScaler using training data only.
    already_standardised = all(
        abs(X[column].mean()) < 0.10 and 0.80 < X[column].std() < 1.20
        for column in FEATURES
    )

    if already_standardised:
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        scaler = None
        scaling_note = "Features appeared already standardised; no additional scaling applied."
    else:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=FEATURES, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=FEATURES, index=X_test.index
        )
        scaling_note = "StandardScaler fitted on the training set and applied to the test set."

    # Elbow Method
    k_values = range(1, 11)
    inertias = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        model.fit(X_train_scaled)
        inertias.append(model.inertia_)

    elbow_df = pd.DataFrame({"k": list(k_values), "inertia": inertias})
    elbow_df.to_csv(OUTPUT_DIR / "elbow_values.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_values), inertias, marker="o")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.xticks(list(k_values))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "elbow_method.png", dpi=200)
    plt.close()

    # Train final K-Means model
    kmeans = KMeans(n_clusters=SELECTED_K, random_state=RANDOM_STATE, n_init=10)
    train_labels = kmeans.fit_predict(X_train_scaled)
    test_labels = kmeans.predict(X_test_scaled)

    # Add labels to the complete dataset
    if scaler is None:
        X_all_scaled = X.copy()
    else:
        X_all_scaled = pd.DataFrame(
            scaler.transform(X), columns=FEATURES, index=X.index
        )
    df["Cluster"] = kmeans.predict(X_all_scaled)

    # Cluster summary
    summary = df.groupby("Cluster").agg(
        Customer_Count=("Cluster", "size"),
        Average_Tenure=("tenure", "mean"),
        Average_MonthlyCharges=("MonthlyCharges", "mean"),
    ).reset_index()

    if "Churn" in df.columns:
        churn_numeric = df["Churn"].map({"Yes": 1, "No": 0})
        if churn_numeric.notna().any():
            df["Churn_Numeric"] = churn_numeric
            churn_summary = df.groupby("Cluster")["Churn_Numeric"].mean().mul(100)
            summary["Churn_Rate_Percent"] = summary["Cluster"].map(churn_summary)

    summary.to_csv(OUTPUT_DIR / "cluster_summary.csv", index=False)
    df.to_csv(OUTPUT_DIR / "clustered_dataset.csv", index=False)

    # Cluster visualisation
    plt.figure(figsize=(9, 6))
    for cluster_id in sorted(df["Cluster"].unique()):
        points = df[df["Cluster"] == cluster_id]
        plt.scatter(
            points["tenure"], points["MonthlyCharges"],
            s=15, alpha=0.55, label=f"Cluster {cluster_id}"
        )
    centers = kmeans.cluster_centers_
    if scaler is not None:
        centers = scaler.inverse_transform(centers)
    plt.scatter(centers[:, 0], centers[:, 1], marker="X", s=180, label="Centroids")
    plt.xlabel("Tenure")
    plt.ylabel("Monthly Charges")
    plt.title("Customer Clusters")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cluster_visualisation.png", dpi=200)
    plt.close()

    # Save model and scaler information
    joblib.dump(
        {"model": kmeans, "scaler": scaler, "features": FEATURES},
        OUTPUT_DIR / "kmeans_model.pkl",
    )

    with open(OUTPUT_DIR / "README.txt", "w", encoding="utf-8") as file:
        file.write("Stage 2 Customer Clustering Analysis\n")
        file.write(f"Records analysed: {len(df)}\n")
        file.write(f"Training records: {len(X_train)}\n")
        file.write(f"Testing records: {len(X_test)}\n")
        file.write(f"Selected K: {SELECTED_K}\n")
        file.write(scaling_note + "\n")
        file.write("Clustering features: tenure and MonthlyCharges\n")
        file.write("Churn was used only for interpretation when available, not for clustering.\n")

    print("Analysis completed successfully.")
    print(f"Records analysed: {len(df)}")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Selected K: {SELECTED_K}")
    print(scaling_note)
    print(f"Outputs saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
