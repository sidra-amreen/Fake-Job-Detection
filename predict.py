"""Check a job posting for signs of fraud.

Usage:
    python predict.py                  # runs 3 example postings
    python predict.py posting.json     # JSON file with the posting fields
"""
import json
import sys
import joblib
import numpy as np
import pandas as pd

from features import LABELS, NUMERIC, build_text, find_red_flags

bundle = joblib.load("job_model.joblib")
model, explainer, threshold = bundle["model"], bundle["explainer"], bundle["threshold"]

DEFAULTS = dict(title="", company_profile="", description="", requirements="",
                has_company_logo=0, has_company_profile=0, telecommuting=0,
                salary_given=0, free_email_contact=0, education_listed=0)


def top_terms(row: pd.DataFrame, k=5):
    """Words/features pushing the score toward 'fake' in this posting (from logistic regression)."""
    pre, lr = explainer.named_steps["pre"], explainer.named_steps["clf"]
    vec = pre.transform(row)
    contrib = vec.multiply(lr.coef_[0]).toarray()[0] if hasattr(vec, "multiply") else vec[0] * lr.coef_[0]
    names = pre.get_feature_names_out()
    idx = [i for i in np.argsort(contrib)[::-1][:k] if contrib[i] > 0.05]
    out = []
    for i in idx:
        n = names[i].split("__")[1]
        if n in LABELS:      # structured feature: show the actual value, not just the name
            v = row[n].iloc[0]
            out.append(f"{LABELS[n]}: {v}" if n == "description_length" else f"{LABELS[n]}: {'yes' if v else 'no'}")
        else:
            out.append(f"'{n}'")
    return out


def assess(posting: dict):
    d = {**DEFAULTS, **posting}
    d["description_length"] = len(d["description"])
    df = pd.DataFrame([d])
    df["text"] = build_text(df)
    row = df[["text"] + NUMERIC]
    p = float(model.predict_proba(row)[0, 1])
    verdict = "LIKELY FAKE" if p >= threshold else ("SUSPICIOUS" if p >= threshold / 2 else "LIKELY LEGIT")
    print(f"Fraud probability: {p:.1%}  ->  {verdict}")
    flags = find_red_flags(df["text"].iloc[0])
    print("  Red flags   :", "; ".join(flags) if flags else "none found")
    print("  Model signals:", ", ".join(top_terms(row)) or "none")


EXAMPLES = {
    "Obvious scam": dict(
        title="Data Entry Assistant", description="URGENT hiring! Earn $700 per week working from home. "
        "No experience needed. A small registration fee is required. Contact us on WhatsApp or "
        "jobs.hiring.now@gmail.com", requirements="Anyone can apply.", telecommuting=1, free_email_contact=1),
    "Subtle scam": dict(
        title="Customer Support Agent", company_profile="Leading provider of innovative solutions.",
        description="We are looking for a Customer Support Agent with strong communication skills. "
        "Collaborate with cross-functional teams. Interviews on Telegram. Equipment must be purchased first "
        "and reimbursed after your first week.", requirements="Experience with customer service.",
        has_company_profile=1, telecommuting=1),
    "Legitimate posting": dict(
        title="Data Analyst", company_profile="Orbit Analytics is an established firm with 850 employees.",
        description="We are looking for a Data Analyst with strong SQL and Excel skills. Prepare weekly reports and "
        "present insights to leadership. We offer health insurance and a learning budget.",
        requirements="3+ years of experience with SQL and Python.", has_company_logo=1, has_company_profile=1,
        salary_given=1, education_listed=1),
}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        assess(json.load(open(sys.argv[1])))
    else:
        for name, s in EXAMPLES.items():
            print(f"\n== {name} ==")
            assess(s)
