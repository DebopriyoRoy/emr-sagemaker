"""
predict.py — Universal Fraud Detection Inference Script
========================================================
Run predictions on any new CSV file using the deployed
SageMaker endpoint universal-fraud-endpoint.

Usage:
    python predict.py \
        --csv s3://your-bucket/your_data.csv \
        --amount_col Amount \
        --time_col timestamp \
        --user_col sender_account \
        --output s3://your-bucket/predictions.csv
"""

import argparse
import json
import numpy as np
import pandas as pd
import boto3
import joblib

ENDPOINT_NAME = 'universal-fraud-endpoint'
REGION        = 'us-east-1'

FEATURE_COLS = [
    'log_amount', 'amount_z', 'is_high_amount',
    'hour', 'hour_sin', 'hour_cos', 'is_night',
    'day_of_week', 'is_weekend', 'txn_month',
    'user_mean_amount', 'user_std_amount',
    'user_txn_count', 'user_amount_z',
    'txn_count_5m',
    'pca_l2', 'pca_max_abs', 'pca_mean'
]

def extract_features(raw_df, amount_col, time_col=None,
                     time_is_numeric=False, user_col=None):
    feat = pd.DataFrame(index=raw_df.index)
    amt  = pd.to_numeric(raw_df[amount_col], errors='coerce').fillna(0)

    feat['log_amount']     = np.log1p(amt)
    feat['amount_z']       = (amt - amt.mean()) / (amt.std() or 1)
    feat['is_high_amount'] = (amt > amt.quantile(0.95)).astype(int)

    if time_col and time_col in raw_df.columns:
        if time_is_numeric:
            hour = (pd.to_numeric(raw_df[time_col]) / 3600) % 24
            feat['day_of_week'] = 0
            feat['is_weekend']  = 0
            feat['txn_month']   = 0
        else:
            dt   = pd.to_datetime(raw_df[time_col], errors='coerce')
            hour = dt.dt.hour
            feat['day_of_week'] = dt.dt.dayofweek
            feat['is_weekend']  = (dt.dt.dayofweek >= 5).astype(int)
            feat['txn_month']   = dt.dt.month
        feat['hour']     = hour
        feat['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        feat['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        feat['is_night'] = ((hour >= 22) | (hour <= 6)).astype(int)
    else:
        for c in ['hour','hour_sin','hour_cos','is_night',
                  'day_of_week','is_weekend','txn_month']:
            feat[c] = 0

    if user_col and user_col in raw_df.columns:
        us = raw_df.groupby(user_col)[amount_col].agg(
            user_mean_amount='mean',
            user_std_amount='std',
            user_txn_count='count'
        ).reset_index()
        m = raw_df[[user_col, amount_col]].merge(us, on=user_col, how='left')
        feat['user_mean_amount'] = m['user_mean_amount'].values
        feat['user_std_amount']  = m['user_std_amount'].fillna(1).values
        feat['user_txn_count']   = m['user_txn_count'].values
        feat['user_amount_z']    = (amt - m['user_mean_amount'].values) / m['user_std_amount'].fillna(1).values
    else:
        feat['user_mean_amount'] = 0.0
        feat['user_std_amount']  = 0.0
        feat['user_txn_count']   = 0
        feat['user_amount_z']    = 0.0

    feat['txn_count_5m'] = 0

    v_cols = [c for c in raw_df.columns if c.startswith('V') and c[1:].isdigit()]
    if v_cols:
        v = raw_df[v_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        feat['pca_l2']      = np.sqrt((v**2).sum(axis=1))
        feat['pca_max_abs'] = v.abs().max(axis=1)
        feat['pca_mean']    = v.mean(axis=1)
    else:
        feat['pca_l2']      = 0.0
        feat['pca_max_abs'] = 0.0
        feat['pca_mean']    = 0.0

    return feat.reindex(columns=FEATURE_COLS, fill_value=0).fillna(0)


def predict(csv_path, amount_col, time_col=None,
            time_is_numeric=False, user_col=None,
            output_path=None, threshold=0.5):

    so = {'anon': False} if csv_path.startswith('s3://') else {}
    df = pd.read_csv(csv_path, storage_options=so)
    print(f'Loaded: {df.shape}')

    feat    = extract_features(df, amount_col, time_col, time_is_numeric, user_col)
    runtime = boto3.client('sagemaker-runtime', region_name=REGION)

    all_probas, all_preds = [], []
    for i in range(0, len(feat), 100):
        batch  = feat.iloc[i:i+100].to_dict(orient='records')
        resp   = runtime.invoke_endpoint(
            EndpointName = ENDPOINT_NAME,
            ContentType  = 'application/json',
            Body         = json.dumps(batch)
        )
        result = json.loads(resp['Body'].read())
        all_probas.extend(result['probabilities'])
        all_preds.extend(result['predictions'])

    df['fraud_probability'] = all_probas
    df['fraud_predicted']   = all_preds

    print(f'Total: {len(df)} | Fraud detected: {sum(all_preds)} ({sum(all_preds)/len(all_preds):.2%})')

    if output_path:
        so_out = {'anon': False} if output_path.startswith('s3://') else {}
        df.to_csv(output_path, index=False, storage_options=so_out)
        print(f'Saved: {output_path}')

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',             required=True)
    parser.add_argument('--amount_col',      required=True)
    parser.add_argument('--time_col',        default=None)
    parser.add_argument('--time_is_numeric', action='store_true')
    parser.add_argument('--user_col',        default=None)
    parser.add_argument('--output',          default=None)
    parser.add_argument('--threshold',       type=float, default=0.5)
    args = parser.parse_args()

    predict(
        csv_path        = args.csv,
        amount_col      = args.amount_col,
        time_col        = args.time_col,
        time_is_numeric = args.time_is_numeric,
        user_col        = args.user_col,
        output_path     = args.output,
        threshold       = args.threshold
    )
