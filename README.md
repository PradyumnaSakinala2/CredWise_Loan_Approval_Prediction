# CredWise — Loan Approval Prediction

Predicting whether a loan application is approved, from applicant financials and
demographics. The project compares three classifiers on a 1,000-row dataset and
tests whether squaring the two strongest predictors improves separation.

**Precision is the metric that matters here.** A false positive is an approved loan
that should have been declined — a direct credit loss — so the model is tuned to be
right when it says *approve*, even at some cost to recall.

---

## Results

200-row held-out test set (`test_size=0.2`, `random_state=42`).

### Baseline feature set

| Model | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.783 | 0.770 | 0.777 | 0.865 |
| KNN (k=5) | 0.627 | 0.525 | 0.571 | 0.760 |
| **Gaussian Naive Bayes** | **0.804** | 0.738 | 0.769 | 0.865 |

### Engineered feature set
`Credit_Score` and `DTI_Ratio` replaced by their squared transforms.

| Model | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| **Logistic Regression** | **0.790** | **0.803** | **0.797** | **0.875** |
| KNN (k=5) | 0.620 | 0.508 | 0.559 | 0.755 |
| Gaussian Naive Bayes | 0.783 | 0.770 | 0.777 | 0.865 |

**Takeaway.** Naive Bayes wins on raw precision (0.804); Logistic Regression on the
engineered set is the better all-round model, gaining 7.8 points of recall and 2.8 of
F1 for 1.4 points of precision. KNN is clearly unsuited — distance metrics degrade in
the ~25-dimensional one-hot space.

Caveat worth stating plainly: with 200 test rows, a 2-point gap is roughly 4
applicants. The Naive Bayes / Logistic Regression difference is **not**
statistically meaningful on a single split. See *Known limitations*.

---

## Dataset

1,000 applications, 20 columns, target `Loan_Approved` (Yes/No). Roughly 70/30
declined-to-approved, so accuracy alone is a misleading headline.

| Group | Columns |
|---|---|
| Income | `Applicant_Income`, `Coapplicant_Income` |
| Credit | `Credit_Score`, `Existing_Loans`, `DTI_Ratio` |
| Assets | `Savings`, `Collateral_Value` |
| Loan terms | `Loan_Amount`, `Loan_Term`, `Loan_Purpose` |
| Demographics | `Age`, `Gender`, `Marital_Status`, `Dependents`, `Education_Level` |
| Context | `Employment_Status`, `Employer_Category`, `Property_Area` |

---

## Pipeline

1. **Imputation** — mean for numeric columns, mode for categorical.
2. **Encoding** — label-encode `Education_Level` and the target; one-hot the six
   nominal columns with `drop="first"`.
3. **Split and scale** — 80/20 split, then `StandardScaler` fit on the training
   fold only.
4. **Train and evaluate** — Logistic Regression, KNN, Gaussian Naive Bayes.
5. **Feature engineering** — repeat on squared `Credit_Score` / `DTI_Ratio`.

---

## Running it

```bash
pip install -r requirements.txt

# reproducible script version — prints every metric table above
python src/train.py

# or step through the exploratory analysis and plots
jupyter notebook notebooks/credit_wise.ipynb
```

```
credwise/
├── data/loan_approval_data.csv     # 1,000 applications
├── notebooks/credit_wise.ipynb     # EDA, plots, original analysis
├── src/train.py                    # reproducible end-to-end pipeline
└── requirements.txt
```

---

## Known limitations

Honest notes on what this project does *not* yet establish:

- **Single-split evaluation.** All numbers come from one 80/20 split. Differences
  under ~5 points are inside the noise band. Cross-validation is the fix.
- **Imputation leakage.** Means and modes are computed over the full dataset before
  splitting, so test-fold statistics leak into training. The effect is small for
  mean imputation but it inflates results. Correct fix: wrap imputation, encoding
  and scaling in a `sklearn.pipeline.Pipeline` fit inside each fold.
- **No class-imbalance handling.** With a ~70/30 split, `class_weight="balanced"`
  or threshold tuning on the predicted probability would likely beat picking a
  different model family.
- **Protected attributes retained.** `Gender` and `Marital_Status` are used as
  features. For any real credit-scoring use this is a fairness and compliance
  problem, not just a modelling choice — it warrants a disparate-impact check at
  minimum.
- **Notebook reproducibility.** The notebook's stored engineered-feature outputs
  were produced in a session with out-of-order and unsaved cell executions, so they
  differ by ~1 point from `src/train.py`. The script is the authoritative record.

## Next steps

- Replace manual preprocessing with a `Pipeline` + `StratifiedKFold`
- Add tree ensembles (Random Forest, XGBoost) as stronger baselines
- Tune the decision threshold against an explicit cost ratio for false approvals
- Report feature importance / SHAP values for adverse-action explainability

---

## License

MIT — see [LICENSE](LICENSE).
