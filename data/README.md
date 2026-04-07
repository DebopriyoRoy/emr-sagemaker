# Data

All datasets are stored in Amazon S3 and are **not committed to this repository**.

## Input Datasets (S3)

| Dataset | S3 Path | Size | Rows |
|---|---|---|---|
| financial_fraud_detection_dataset.csv | `s3://databucket-fraud-detection/input_data/financial_fraud_detection_dataset.csv` | 759 MB | 5,000,000 |
| creditcard.csv | `s3://databucket-fraud-detection/input_data/creditcard.csv` | 143 MB | 284,807 |
| credit_card_fraud_dataset.csv | `s3://databucket-fraud-detection/input_data/credit_card_fraud_dataset.csv` | Variable | 100,000 |
| mixed_fraud_test_dataset_1M.csv | `s3://databucket-fraud-detection/input_data/mixed_fraud_test_dataset_1M.csv` | 175 MB | 1,000,000 |

## EMR Output (Universal Features)

| Partition | S3 Path | Rows |
|---|---|---|
| source=financial | `s3://databucket-fraud-detection/output_data/universal-features/source=financial/` | 5,000,000 |
| source=creditcard | `s3://databucket-fraud-detection/output_data/universal-features/source=creditcard/` | 284,807 |
| source=credit_card_fraud | `s3://databucket-fraud-detection/output_data/universal-features/source=credit_card_fraud/` | 100,000 |

## Model Artifacts (S3)

| Artifact | Primary S3 | DR S3 |
|---|---|---|
| model.tar.gz | `s3://databucket-fraud-detection/models/universal-fraud-xgb/` | `s3://fraud-detection-dr/models/universal-fraud-xgb/` |
| metrics.json | `s3://databucket-fraud-detection/results/universal-fraud-xgb/` | `s3://fraud-detection-dr/results/universal-fraud-xgb/` |
