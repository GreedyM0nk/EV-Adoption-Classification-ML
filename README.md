# EV Adoption Classification

## a. Problem statement

This project predicts whether a buyer will purchase an electric vehicle (`Will_Buy_EV`)
from demographic, income, commute, vehicle ownership, charging-access, environmental
concern, subsidy, and range-anxiety information. The goal is to compare five standard
classification algorithms using a consistent, leakage-safe evaluation protocol and make
the results available through an interactive Streamlit dashboard.

## b. Dataset description

The dataset is the Kaggle [EV Adoption Behavior and Range Anxiety
Dataset](https://www.kaggle.com/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety).
It contains 10,000 buyer records, 14 candidate input columns, and the binary target
`Will_Buy_EV`. The target contains 8,250 `No` records and 1,750 `Yes` records, so the
dataset is imbalanced and accuracy is not sufficient by itself.

`Buyer_ID` is an identifier and is excluded from model features. The remaining features
include numeric and categorical values. Missing values occur in `Annual_Income_USD`,
`Daily_Commute_km`, and `Environmental_Concern_Level`; the training pipelines impute
these values using statistics learned from the training partition only.

The data is split once with a stratified 80/20 holdout and `random_state=42`. All
preprocessing is inside each saved sklearn pipeline. Model selection uses a modest
three-fold stratified grid search on the training partition only. The final metrics
below are calculated on the untouched 20% holdout.

## c. GitHub Repository Link

**Repository:** Replace this line with the public GitHub URL after publishing this
repository.

## d. Models used and evaluation

All models use the same feature set and train/test split. AUC uses the positive `Yes`
class score; SVM uses its calibrated probability output.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8175 | 0.9129 | 0.4879 | 0.8657 | 0.6241 | 0.5527 |
| Decision Tree | 0.8060 | 0.8972 | 0.4705 | 0.8657 | 0.6097 | 0.5359 |
| KNN | 0.8575 | 0.8705 | 0.6650 | 0.3743 | 0.4790 | 0.4262 |
| Gaussian Naive Bayes | 0.6380 | 0.8806 | 0.3141 | 0.9029 | 0.4661 | 0.3683 |
| Support Vector Machine | 0.8180 | 0.9143 | 0.4887 | 0.8686 | 0.6255 | 0.5548 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong linear baseline with high AUC and high recall. Its balanced class weighting detects most likely adopters, although precision is lower because the positive class is relatively uncommon. |
| Decision Tree | Captures nonlinear rules and reaches the same recall as logistic regression, but has slightly weaker AUC, F1, and MCC on this holdout. |
| KNN | Has the highest accuracy and precision, but its recall is much lower. It misses many positive adopters, so its accuracy is partly influenced by the majority class. |
| Gaussian Naive Bayes | Produces the highest recall but the lowest accuracy, precision, F1, and MCC. Its conditional-independence assumption appears too permissive for this feature mix. |
| Support Vector Machine | Provides the best overall balance on this split: highest AUC, F1, and MCC, with recall close to the best models. |
| **Overall winner** | **Support Vector Machine** by the combined AUC, F1, and MCC criteria. Logistic Regression is a very close, more interpretable alternative. The winner is selected from holdout evidence, not accuracy alone. |

## Repository contents

```text
app.py                         Streamlit evaluation dashboard
train.py                       Reproducible training and evaluation script
requirements.txt               Runtime dependencies
test_data.csv                 The fixed labeled 20% holdout used by the app
model/
  logistic_regression.joblib
  decision_tree.joblib
  knn.joblib
  gaussian_nb.joblib
  svm.joblib
  metadata.json
  metrics.json
EV_Adoption_and_Range_Anxiety_Dataset.csv
```

## Run locally

1. Create and activate a Python environment.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Rebuild the saved models and test split:

   ```bash
   python train.py
   ```

4. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

5. Upload `test_data.csv`, choose a model, and review the six metrics, confusion matrix,
   classification report, and downloadable predictions. The app loads saved pipelines
   and never retrains against uploaded data.

## Deployment checklist

- Push `app.py`, `train.py`, `requirements.txt`, `README.md`, `test_data.csv`, and the
  complete `model/` directory to a public GitHub repository.
- In Streamlit Community Cloud, select the repository, the deployment branch, and
  `app.py` as the entry point.
- Verify that the public URL opens without authentication and that a test CSV can be
  uploaded successfully.
- Replace the repository placeholder above with the actual clickable GitHub URL.
- Include the public Streamlit URL, the BITS Virtual Lab execution screenshot, and this
  README content in the final submission PDF in the assignment's required order.

## Academic integrity and reproducibility

The Kaggle source is cited above, preprocessing is fitted only on training data, and
the random seed and evaluation protocol are explicit. The dashboard text and layout
are customized for this dataset. Commit the work in meaningful increments and explain
the results in your own words before submission.