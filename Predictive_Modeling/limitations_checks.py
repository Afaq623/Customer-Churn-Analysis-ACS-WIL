"""
Limitation checks - evidence used in the Limitations and Proposed Solutions document.

  1. Duplicate rows and train/test overlap, and how much they affect the test result
  2. 5-fold cross-validation of the 16-8 ANN on all 7,043 customers
  3. Stability of the K-means segments across random starts

Inputs : churn_ann_model.h5, data/churn_train_processed.csv, data/churn_test_processed.csv,
         data/Dataset_ATS_v2.csv, data/churn_clustering_ready.csv
Output : limitations_check_results.csv   (run time: a few minutes)
"""
import os
from pathlib import Path

# Your project folder. If it exists it is used; otherwise the script's own folder is used.
# (Set to None before sharing the code publicly, so your username is not exposed.)
PROJECT_DIR = r"C:\Users\neesh\OneDrive\Desktop\WilProgram\Final"
HERE = Path(PROJECT_DIR) if PROJECT_DIR and Path(PROJECT_DIR).is_dir() else Path(__file__).resolve().parent
os.chdir(HERE)
RESULTS = []


def log(check, metric, value):
    RESULTS.append((check, metric, value))
    print(f"{check} | {metric}: {value}")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd, random, warnings; warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

tr=pd.read_csv('data/churn_train_processed.csv'); te=pd.read_csv('data/churn_test_processed.csv'); raw=pd.read_csv('data/Dataset_ATS_v2.csv')
F=[c for c in tr.columns if c!='Churn']
log("duplicates", "raw duplicate rows", int(raw.duplicated().sum()))
log("duplicates", "train duplicate rows", int(tr.duplicated().sum())); log("duplicates", "test duplicate rows", int(te.duplicated().sum()))
m=te.merge(tr.drop_duplicates(), on=list(tr.columns), how='left', indicator=True)
log("overlap", "test rows identical to a training row", f"{int((m['_merge']=='both').sum())} of {len(te)}")

def fit(Xtr,ytr,seed=42):
    tf.keras.utils.set_random_seed(seed)
    Xa,Xv,ya,yv=train_test_split(Xtr,ytr,test_size=.2,stratify=ytr,random_state=42)
    mdl=keras.Sequential([layers.Input(shape=(Xtr.shape[1],)),layers.Dense(16,activation='relu'),layers.Dense(8,activation='relu'),layers.Dense(1,activation='sigmoid')])
    mdl.compile(optimizer='adam',loss='binary_crossentropy')
    mdl.fit(Xa,ya,validation_data=(Xv,yv),epochs=50,batch_size=32,callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)],verbose=0)
    return mdl
def ev(mdl,X,y):
    p=mdl.predict(X,verbose=0).ravel(); return dict(AUC=roc_auc_score(y,p),Acc=accuracy_score(y,p>=.5),Rec=recall_score(y,p>=.5),Prec=precision_score(y,p>=.5))

# 1. de-duplication sensitivity
trd=tr.drop_duplicates()
ted=m[m['_merge']=='left_only'].drop(columns='_merge').drop_duplicates()
log("de-duplication", "clean train / clean test rows", f"{len(trd)} / {len(ted)}")
orig=keras.models.load_model('churn_ann_model.h5',compile=False)
log("de-duplication", "original model on clean test", {k:round(v,3) for k,v in ev(orig,ted[F].values.astype("float32"),ted.Churn.values).items()})
new=fit(trd[F].values.astype('float32'),trd.Churn.values)
log("de-duplication", "retrained on de-duplicated data, clean test", {k:round(v,3) for k,v in ev(new,ted[F].values.astype("float32"),ted.Churn.values).items()})

# 2. 5-fold cross-validation on all 7,043 customers
al=pd.concat([tr,te],ignore_index=True); X=al[F].values.astype('float32'); y=al.Churn.values
aucs=[]
for k,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y)):
    aucs.append(ev(fit(X[a],y[a]),X[b],y[b])['AUC'])
log("cross-validation", "5-fold AUC per fold", [float(x) for x in np.round(aucs,4)]); log("cross-validation", "mean AUC / std", f"{np.mean(aucs):.4f} / {np.std(aucs,ddof=1):.4f}")

# 3. cluster stability across random starts
cl=pd.read_csv('data/churn_clustering_ready.csv')[['tenure','MonthlyCharges']].values
ref=KMeans(4,n_init=10,random_state=42).fit_predict(cl)
aris=[adjusted_rand_score(ref,KMeans(4,n_init=10,random_state=s).fit_predict(cl)) for s in range(1,11)]
log("cluster stability", "ARI vs seed 42 (10 seeds): min / mean", f"{min(aris):.4f} / {np.mean(aris):.4f}")

# 4. How much better can other models do with the same data? (ceiling check)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
lr_all = LogisticRegression(max_iter=1000).fit(tr[F], tr.Churn)
log("ceiling", "logistic regression, all 12 inputs, test AUC", round(roc_auc_score(te.Churn, lr_all.predict_proba(te[F])[:, 1]), 4))
F4 = ["tenure", "MonthlyCharges", "Dependents", "SeniorCitizen"]
lr_4 = LogisticRegression(max_iter=1000).fit(tr[F4], tr.Churn)
log("ceiling", "logistic regression, 4 inputs only, test AUC", round(roc_auc_score(te.Churn, lr_4.predict_proba(te[F4])[:, 1]), 4))
g = [roc_auc_score(te.Churn, HistGradientBoostingClassifier(random_state=s, max_iter=150, learning_rate=0.05).fit(tr[F], tr.Churn).predict_proba(te[F])[:, 1]) for s in range(3)]
log("ceiling", "gradient-boosted trees, 12 inputs, test AUC (mean of 3 seeds)", round(float(np.mean(g)), 4))

import csv
with open("limitations_check_results.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["check", "metric", "value"]); w.writerows(RESULTS)
print("Saved limitations_check_results.csv")
