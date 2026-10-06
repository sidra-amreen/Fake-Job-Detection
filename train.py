import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             average_precision_score, classification_report,
                             precision_recall_curve)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from features import NUMERIC, build_text
from generate_data import make

DATA = "data/jobs.csv"


def make_pipe(clf):
    pre = ColumnTransformer([
        ("text", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=20000), "text"),
        ("num", StandardScaler(), NUMERIC),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])


def main():
    if not os.path.exists(DATA):
        os.makedirs("data", exist_ok=True)
        make().to_csv(DATA, index=False)
        print("Generated synthetic dataset -> data/jobs.csv")
    df = pd.read_csv(DATA)
    df["text"] = build_text(df)
    X, y = df[["text"] + NUMERIC], df["fraudulent"]
    print(f"Postings: {len(df)} | fake rate: {y.mean():.1%}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "Logistic Regression": make_pipe(LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Linear SVM (calibrated)": make_pipe(CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=20000), cv=3)),
        "Random Forest": make_pipe(RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
    }

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    fitted, best_name, best_ap = {}, None, -1
    print("\nPR-AUC (average precision): 5-fold CV | test   (accuracy is misleading with imbalance)")
    for name, m in models.items():
        cv_ap = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="average_precision").mean()
        m.fit(X_tr, y_tr)
        te_ap = average_precision_score(y_te, m.predict_proba(X_te)[:, 1])
        fitted[name] = m
        print(f"  {name:<24} {cv_ap:.3f} | {te_ap:.3f}")
        if cv_ap > best_ap:
            best_name, best_ap = name, cv_ap

    best = fitted[best_name]

    oof = cross_val_predict(make_pipe(models[best_name].named_steps["clf"]), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    p, r, t = precision_recall_curve(y_tr, oof)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-9)
    threshold = float(t[np.argmax(f1)])

    proba = best.predict_proba(X_te)[:, 1]
    pred = (proba >= threshold).astype(int)
    print(f"\nBest model: {best_name} | threshold chosen via CV = {threshold:.2f}\n")
    print(classification_report(y_te, pred, target_names=["legit", "fake"]))

    # Plots
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for name, m in fitted.items():
        PrecisionRecallDisplay.from_estimator(m, X_te, y_te, ax=ax[0], name=name)
    ax[0].set_title("Precision-Recall curves")
    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["legit", "fake"], cmap="Blues", ax=ax[1])
    ax[1].set_title(f"Confusion matrix - {best_name}")

    lr = fitted["Logistic Regression"]
    names = lr.named_steps["pre"].get_feature_names_out()
    coef = lr.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    sel = np.concatenate([order[:8], order[-8:]])
    ax[2].barh([n.split("__")[1] for n in names[sel]], coef[sel], color=["#59A14F"] * 8 + ["#E15759"] * 8)
    ax[2].set_title("Top signals: green = legit, red = fake")
    plt.tight_layout()
    plt.savefig("results.png", dpi=120)

    joblib.dump({"model": best, "explainer": lr, "threshold": threshold, "name": best_name}, "job_model.joblib")
    print("Saved job_model.joblib and results.png")


if __name__ == "__main__":
    main()
