# Customer Churn Prediction Using an Artificial Neural Network

Stage 3 project: predicting telecommunications customer churn with an ANN, segmenting customers, identifying churn drivers, and recommending retention strategies.

## Overview

The project uses `Dataset_ATS_v2.csv` (7,043 customers, 26.5% churn) and has four parts:

1. **Customer segmentation.** K-means on tenure and monthly charges (k = 4, chosen with the elbow method and silhouette score).
2. **ANN churn model.** A feed-forward network (12 inputs, 16 ReLU, 8 ReLU, 1 sigmoid output; 353 trainable parameters) trained with Adam and binary cross-entropy.
3. **Churn driver analysis.** Permutation importance, logistic-regression odds ratios, churn rates and chi-square tests.
4. **Limitation checks.** Duplicates, cross-validation, cluster stability and a model-ceiling comparison.

## Headline results

**Model performance on 1,409 unseen test customers**

| Metric | Threshold 0.50 | Tuned threshold 0.39 |
|---|---|---|
| Accuracy | 78.2% | 76.7% |
| Precision | 60.7% | 55.2% |
| Recall | 50.8% | 63.9% |
| F1 score | 55.3% | 59.2% |
| AUC-ROC | 80.5% | 80.5% |

The threshold of 0.39 was chosen on validation data, not on the test set. Retraining with five random seeds gave an AUC of 80.6% (spread 0.1 points). On a test set with duplicates and training-set twins removed, the AUC is 79.1%, so about 79% is a cautious estimate for genuinely new customers.

**Customer segments**

| Segment | Customers | Avg tenure (months) | Avg monthly charge | Churn rate | Share of all churners |
|---|---|---|---|---|---|
| New Premium (high risk) | 2,202 | 14.8 | $81.23 | 49.2% | 57.9% |
| New Budget | 1,733 | 10.5 | $32.49 | 24.5% | 22.7% |
| Loyal Premium | 1,948 | 58.8 | $93.08 | 15.7% | 16.4% |
| Loyal Budget | 1,160 | 54.1 | $34.15 | 4.8% | 3.0% |

**Main findings**

- Tenure and monthly charges drive churn. First-year customers paying $70 to $95 per month churn at 67%; customers with more than four years of tenure paying under $40 churn at 1.5%.
- Senior citizens (41.7% against 23.6%) and customers without dependents (31.3% against 15.4%) also churn more.
- Contract type, internet service, multiple lines and gender show no relationship with churn in this dataset.
- The ANN performs the same as a logistic regression (AUC about 0.81), so the data, not the model, limits accuracy.
- The model catches 83% of leavers in the New Premium segment but rarely detects loyal customers who leave.

## Repository structure

```
Predictive_Modeling/
├── ANN_Architecture.docx          Architecture document (deliverable 3-1a)
├── churn_ann_model.h5             Trained ANN model (deliverable 3-1b)
├── churn_prediction.py            Builds, trains, evaluates and saves the model (3-1b, 3-1c)
├── customer_segmentation.py       K-means customer segmentation
├── churn_drivers.py               Churn driver analysis
├── limitations_checks.py          Evidence for the limitations document
├── make_architecture_diagram.py   Draws architecture_diagram.png
├── requirements.txt               Python packages needed
├── predictions.csv                Actual churn, predicted probability and class for each test customer
├── evaluation_results.csv         Accuracy, precision, recall, F1, AUC at both thresholds
├── model_comparison.csv           ANN 16-8 against ANN 64-32 and logistic regression
├── robustness_results.csv         Results for five random seeds
├── cluster_profiles.csv           Segment profiles
├── feature_importance.csv, churn_rate_by_factor.csv, statistical_tests.csv,
│   churn_rate_tenure_x_charges.csv, churn_rate_tenure_x_dependents.csv,
│   limitations_check_results.csv  Analysis tables
├── *.png                          Charts (training curves, confusion matrix, ROC, segments, drivers)
└── data/
    ├── Dataset_ATS_v2.csv                 Raw dataset
    ├── churn_train_processed.csv          Training set (5,634 customers, 80%)
    ├── churn_test_processed.csv           Test set (1,409 customers, 20%)
    ├── churn_clustering_ready.csv         Scaled data used for clustering
    └── customers_with_segments.csv        Every customer with their segment

Final_Report/
├── 1_Summary_of_Key_Findings.docx
├── 2_Churn_and_Retention_Factors.docx
├── 3_Retention_Recommendations.docx
└── 4_Limitations_and_Solutions.docx
```

## Mapping to the Stage 3 deliverables

| Deliverable | Where |
|---|---|
| 3-1: Defined architecture of the ANN | `Predictive_Modeling/ANN_Architecture.docx` |
| 3-1: Trained ANN, with code | `Predictive_Modeling/churn_ann_model.h5`, `churn_prediction.py` |
| 3-1: Predictions and performance evaluation | `predictions.csv`, `evaluation_results.csv`, `confusion_matrix.png`, `roc_curve.png`, `accuracy_loss_graph.png` |
| 3-2: Summary of key findings | `Final_Report/1_Summary_of_Key_Findings.docx` |
| 3-2: Factors contributing to churn and retention | `Final_Report/2_Churn_and_Retention_Factors.docx` |
| 3-2: Targeted retention recommendations | `Final_Report/3_Retention_Recommendations.docx` |
| 3-2: Limitations and proposed solutions | `Final_Report/4_Limitations_and_Solutions.docx` |
| 3-2: Video demonstration | Submitted separately as a video file |

## How to run

Developed and tested with Python 3.13 (Windows) and TensorFlow 2.x; the results were also reproduced on a Linux environment.

```
cd Predictive_Modeling
pip install -r requirements.txt

python customer_segmentation.py     # segments, elbow chart, cluster profiles
python churn_prediction.py          # trains the ANN, writes the model, predictions, metrics and charts
python churn_drivers.py             # needs churn_ann_model.h5, so run after churn_prediction.py
python limitations_checks.py        # optional, takes a few minutes
python make_architecture_diagram.py # optional, redraws the architecture diagram
```

Each script finds its data in the `data` folder next to it and writes its results into the same folder, so it can be run from any directory. `churn_prediction.py` and `limitations_checks.py` take a few minutes because they train several models.

TensorFlow prints some startup warnings (for example about oneDNN and GPU support). They are harmless.

## Reproducibility

Random seeds are fixed. Running the scripts from a clean copy of this repository, with all result files deleted, reproduces the predictions, metrics, importance tables and limitation checks exactly.

## Data notes and limitations

- Only nine customer attributes are available, and only four (tenure, monthly charges, senior citizen, dependents) relate to churn.
- Contract type and internet service show no relationship with churn, which is unusual for telecommunications data. They should be verified with the data provider.
- The dataset has 302 duplicate rows and no customer ID. 93 of the 1,409 test rows are identical to a training row.
- Tenure and monthly charges were standardised using all customers before the train/test split. The scaling does not use the churn label, so the effect is expected to be negligible.
- The findings are associations, not proven causes. Retention strategies should be piloted with a control group.

See `Final_Report/4_Limitations_and_Solutions.docx` for the full list and proposed solutions.
