# Fake Job Detection (Machine Learning)

Flags fraudulent job postings by combining the posting text (TF-IDF) with structured signals
(company logo/profile, salary given, contact email type, remote flag...) and explains why a posting
looks suspicious.

## Setup
```bash
pip install -r requirements.txt
python train.py       # generates data if missing, trains 3 models, saves job_model.joblib + results.png
python predict.py     # checks 3 example postings
python predict.py posting.json   # check your own posting
```

## Pipeline
1. `generate_data.py` - synthetic postings (~8% fake): obvious scams, subtle scams that look professional,
   and legitimate postings that sometimes use words like "remote" or "immediate start", plus 1% label noise.
2. `train.py` - ColumnTransformer (TF-IDF on text + scaled numeric features), compares Logistic Regression,
   calibrated Linear SVM and Random Forest by cross-validated **PR-AUC** (accuracy is misleading when only
   ~8% of postings are fake). The decision threshold is picked on out-of-fold training predictions to
   maximize F1, so the test set stays untouched.
3. `predict.py` - fraud probability, verdict, rule-based red flags (upfront fees, WhatsApp/Telegram
   interviews, free-email contacts, unrealistic pay, etc.) and the strongest model signals.

## Using real data
Download the EMSCAD "Fake Job Postings" CSV from Kaggle and save it as `data/jobs.csv`, renaming columns to
`title, company_profile, description, requirements, has_company_logo, telecommuting, fraudulent` and creating
`has_company_profile, salary_given, free_email_contact, education_listed, description_length` from the
other fields (or edit `NUMERIC` in `features.py`).

## Limitations
Scammers adapt, so retrain regularly. A "LIKELY LEGIT" verdict is not a guarantee: never pay to get a job,
and verify the company through its official website.
