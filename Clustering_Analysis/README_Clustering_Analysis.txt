Stage 2 - Clustering Analysis

Dataset: churn_clustering_ready (2).csv

Clustering variables:
- tenure
- MonthlyCharges

Data preparation:
- The supplied dataset already contains standardised tenure and MonthlyCharges values.
- No additional scaling was applied to these two clustering variables.
- An 80/20 train/test split was created with random_state=42.
- Training rows: 5634
- Testing rows: 1409

Elbow method:
- K values tested: 1 to 10.
- Based on the elbow curve, K=4 was selected as a reasonable working solution.
- Note: the elbow method is a visual/heuristic method; K=4 should be presented as the selected value from the observed curve.

K-Means:
- n_clusters = 4
- random_state = 42
- n_init = 10

Important:
The clustering model is based on tenure and MonthlyCharges only. The Churn column is not used as a clustering input. It is retained only for post-clustering interpretation of churn rates.

Cluster interpretation:
- Cluster 0: relatively low tenure and low monthly charges.
- Cluster 1: relatively high tenure and high monthly charges.
- Cluster 2: relatively low tenure and relatively high monthly charges.
- Cluster 3: relatively high tenure and low monthly charges.

Because the supplied clustering variables are standardised, these descriptions are relative rather than raw-dollar/raw-month values.
