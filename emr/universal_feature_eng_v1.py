"""
Universal Feature Engineering for Fraud Detection
Processes 3 datasets simultaneously on EMR
Output: Single parquet in s3://databucket-fraud-detection/output_data/
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, log1p, lit, sin, cos, floor,
    unix_timestamp, hour, dayofweek, month,
    when, avg, stddev, count, percent_rank,
    to_timestamp, isnan, isnull
)
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, DoubleType
import math

# ── Spark Session ─────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("universal-fraud-feature-eng") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ── S3 Paths ──────────────────────────────────────────────────────────────
BASE       = "s3://databucket-fraud-detection"
INPUT      = f"{BASE}/input_data"
OUTPUT     = f"{BASE}/output_data"

DS1 = f"{INPUT}/financial_fraud_detection_dataset.csv"
DS2 = f"{INPUT}/creditcard.csv"
DS3 = f"{INPUT}/credit_card_fraud_dataset.csv"

# ─────────────────────────────────────────────────────────────────────────
# HELPER: Universal feature extraction
# ─────────────────────────────────────────────────────────────────────────
def add_amount_features(df, amount_col):
    """Add universal amount-based features"""
    # Log transform
    df = df.withColumn("log_amount", log1p(col(amount_col).cast(DoubleType())))

    # Z-score
    stats = df.select(
        avg(amount_col).alias("mean_amt"),
        stddev(amount_col).alias("std_amt")
    ).collect()[0]
    mean_amt = float(stats["mean_amt"] or 0)
    std_amt  = float(stats["std_amt"]  or 1)

    df = df.withColumn("amount_z",
        (col(amount_col).cast(DoubleType()) - lit(mean_amt)) / lit(std_amt)
    )

    # Percentile rank
    w  = Window.orderBy(amount_col)
    df = df.withColumn("amount_pct", percent_rank().over(w))

    # High amount flag (top 5%)
    p95 = df.approxQuantile(amount_col, [0.95], 0.01)[0]
    df  = df.withColumn("is_high_amount",
        when(col(amount_col) > lit(p95), 1).otherwise(0)
    )
    return df


def add_time_features_numeric(df, time_col):
    """For numeric time columns (like creditcard.csv Time in seconds)"""
    df = df.withColumn("hour",
        (col(time_col) / 3600) % 24
    )
    df = df.withColumn("hour_sin",
        sin(lit(2 * math.pi) * col("hour") / lit(24.0))
    )
    df = df.withColumn("hour_cos",
        cos(lit(2 * math.pi) * col("hour") / lit(24.0))
    )
    df = df.withColumn("is_night",
        when((col("hour") >= 22) | (col("hour") <= 6), 1).otherwise(0)
    )
    # No day/month available from numeric time
    df = df.withColumn("day_of_week", lit(0))
    df = df.withColumn("is_weekend",  lit(0))
    df = df.withColumn("txn_month",   lit(0))
    df = df.withColumn("txn_year",    lit(0))
    return df


def add_time_features_datetime(df, time_col):
    """For datetime string columns"""
    df = df.withColumn("ts", unix_timestamp(to_timestamp(col(time_col))))
    df = df.withColumn("hour",        (col("ts") / 3600) % 24)
    df = df.withColumn("hour_sin",    sin(lit(2*math.pi) * col("hour") / lit(24.0)))
    df = df.withColumn("hour_cos",    cos(lit(2*math.pi) * col("hour") / lit(24.0)))
    df = df.withColumn("is_night",    when((col("hour")>=22)|(col("hour")<=6),1).otherwise(0))
    df = df.withColumn("day_of_week", dayofweek(to_timestamp(col(time_col))))
    df = df.withColumn("is_weekend",  when(col("day_of_week").isin([1,7]),1).otherwise(0))
    df = df.withColumn("txn_month",   month(to_timestamp(col(time_col))))
    df = df.withColumn("txn_year",    col("ts").cast(DoubleType()))
    return df


def add_user_features(df, user_col, amount_col):
    """Per-user behavioural features"""
    user_agg = df.groupBy(user_col).agg(
        avg(amount_col).alias("user_mean_amount"),
        stddev(amount_col).alias("user_std_amount"),
        count("*").alias("user_txn_count")
    )
    df = df.join(user_agg, on=user_col, how="left")
    df = df.withColumn("user_amount_z",
        (col(amount_col) - col("user_mean_amount")) /
        when(col("user_std_amount") > 0, col("user_std_amount")).otherwise(1)
    )
    return df


def add_velocity_features(df, time_col_numeric):
    """Transaction velocity in 5-minute buckets"""
    df = df.withColumn("bucket_5m", floor(col(time_col_numeric) / 300))
    bucket_agg = df.groupBy("bucket_5m").count() \
                   .withColumnRenamed("count", "txn_count_5m")
    df = df.join(bucket_agg, on="bucket_5m", how="left")
    return df


def add_pca_features(df, v_cols):
    """PCA magnitude features for V1-V28 columns"""
    from pyspark.sql.functions import sqrt, greatest, abs as sabs

    sq_sum = None
    for c in v_cols:
        term   = col(c).cast(DoubleType()) * col(c).cast(DoubleType())
        sq_sum = term if sq_sum is None else sq_sum + term
    df = df.withColumn("pca_l2", sqrt(sq_sum))

    max_abs = None
    for c in v_cols:
        term    = sabs(col(c).cast(DoubleType()))
        max_abs = term if max_abs is None else greatest(max_abs, term)
    df = df.withColumn("pca_max_abs", max_abs)

    # Mean and std of PCA components
    from pyspark.sql.functions import array, array_min, array_max
    df = df.withColumn("pca_mean",
        sum([col(c).cast(DoubleType()) for c in v_cols]) / lit(len(v_cols))
    )
    return df


# ── UNIVERSAL COLUMNS (final schema) ─────────────────────────────────────
UNIVERSAL_COLS = [
    "log_amount", "amount_z", "amount_pct", "is_high_amount",
    "hour", "hour_sin", "hour_cos", "is_night",
    "day_of_week", "is_weekend", "txn_month",
    "user_mean_amount", "user_std_amount", "user_txn_count", "user_amount_z",
    "txn_count_5m",
    "pca_l2", "pca_max_abs", "pca_mean",
    "label",      # target: 0 or 1
    "source"      # which dataset it came from
]


# ─────────────────────────────────────────────────────────────────────────
# DATASET 1: financial_fraud_detection_dataset.csv
# Columns: transaction_id, timestamp, sender_account, receiver_account,
#          amount, transaction_type, merchant_category, location,
#          device_used, is_fraud, fraud_type, ...
# ─────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Processing Dataset 1: financial_fraud_detection_dataset.csv")
print("=" * 60)

df1 = spark.read.option("header", True).option("inferSchema", True).csv(DS1)
print(f"DS1 shape: {df1.count()} rows x {len(df1.columns)} cols")

# Rename amount column to standard
df1 = df1.withColumnRenamed("amount", "Amount") \
         if "amount" in [c.lower() for c in df1.columns] else df1

amount_col1 = "Amount" if "Amount" in df1.columns else "amount"

df1 = add_amount_features(df1, amount_col1)
df1 = add_time_features_datetime(df1, "timestamp")
df1 = add_user_features(df1, "sender_account", amount_col1)

# Velocity using unix timestamp
df1 = df1.withColumn("bucket_5m",
    floor(unix_timestamp(to_timestamp(col("timestamp"))) / 300)
)
bucket_agg1 = df1.groupBy("bucket_5m").count() \
                  .withColumnRenamed("count", "txn_count_5m")
df1 = df1.join(bucket_agg1, on="bucket_5m", how="left")

# No PCA columns
df1 = df1.withColumn("pca_l2",      lit(0.0))
df1 = df1.withColumn("pca_max_abs", lit(0.0))
df1 = df1.withColumn("pca_mean",    lit(0.0))

# Label
df1 = df1.withColumn("label",
    when(col("is_fraud") == True, 1).otherwise(0).cast(IntegerType())
)
df1 = df1.withColumn("source", lit("financial"))

# Select universal columns only
df1_final = df1.select(*UNIVERSAL_COLS)
print(f"DS1 processed: {df1_final.count()} rows")
print(f"DS1 fraud count: {df1_final.filter(col('label')==1).count()}")


# ─────────────────────────────────────────────────────────────────────────
# DATASET 2: creditcard.csv
# Columns: Time, V1-V28, Amount, Class
# ─────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Processing Dataset 2: creditcard.csv")
print("=" * 60)

df2 = spark.read.option("header", True).option("inferSchema", True).csv(DS2)
print(f"DS2 shape: {df2.count()} rows x {len(df2.columns)} cols")

df2 = add_amount_features(df2, "Amount")
df2 = add_time_features_numeric(df2, "Time")

# No user column
df2 = df2.withColumn("user_mean_amount", lit(0.0))
df2 = df2.withColumn("user_std_amount",  lit(0.0))
df2 = df2.withColumn("user_txn_count",   lit(0))
df2 = df2.withColumn("user_amount_z",    lit(0.0))

# Velocity from Time column
df2 = df2.withColumn("bucket_5m", floor(col("Time") / 300))
bucket_agg2 = df2.groupBy("bucket_5m").count() \
                  .withColumnRenamed("count", "txn_count_5m")
df2 = df2.join(bucket_agg2, on="bucket_5m", how="left")

# PCA features from V1-V28
v_cols = [f"V{i}" for i in range(1, 29)]
df2 = add_pca_features(df2, v_cols)

# Label
df2 = df2.withColumn("label", col("Class").cast(IntegerType()))
df2 = df2.withColumn("source", lit("creditcard"))

df2_final = df2.select(*UNIVERSAL_COLS)
print(f"DS2 processed: {df2_final.count()} rows")
print(f"DS2 fraud count: {df2_final.filter(col('label')==1).count()}")


# ─────────────────────────────────────────────────────────────────────────
# DATASET 3: credit_card_fraud_dataset.csv
# Columns: TransactionID, TransactionDate, Amount, MerchantID,
#          TransactionType, Location, IsFraud
# ─────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Processing Dataset 3: credit_card_fraud_dataset.csv")
print("=" * 60)

df3 = spark.read.option("header", True).option("inferSchema", True).csv(DS3)
print(f"DS3 shape: {df3.count()} rows x {len(df3.columns)} cols")

df3 = add_amount_features(df3, "Amount")
df3 = add_time_features_datetime(df3, "TransactionDate")
df3 = add_user_features(df3, "MerchantID", "Amount")

# Velocity
df3 = df3.withColumn("bucket_5m",
    floor(unix_timestamp(to_timestamp(col("TransactionDate"))) / 300)
)
bucket_agg3 = df3.groupBy("bucket_5m").count() \
                  .withColumnRenamed("count", "txn_count_5m")
df3 = df3.join(bucket_agg3, on="bucket_5m", how="left")

# No PCA columns
df3 = df3.withColumn("pca_l2",      lit(0.0))
df3 = df3.withColumn("pca_max_abs", lit(0.0))
df3 = df3.withColumn("pca_mean",    lit(0.0))

# Label
df3 = df3.withColumn("label", col("IsFraud").cast(IntegerType()))
df3 = df3.withColumn("source", lit("credit_card_fraud"))

df3_final = df3.select(*UNIVERSAL_COLS)
print(f"DS3 processed: {df3_final.count()} rows")
print(f"DS3 fraud count: {df3_final.filter(col('label')==1).count()}")


# ─────────────────────────────────────────────────────────────────────────
# COMBINE ALL 3 DATASETS
# ─────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Combining all 3 datasets...")
print("=" * 60)

combined = df1_final.union(df2_final).union(df3_final)

total = combined.count()
fraud = combined.filter(col("label") == 1).count()

print(f"Combined total : {total}")
print(f"Combined fraud : {fraud}")
print(f"Fraud rate     : {fraud/total:.2%}")


# ─────────────────────────────────────────────────────────────────────────
# WRITE OUTPUT
# ─────────────────────────────────────────────────────────────────────────
print("=" * 60)
print(f"Writing to {OUTPUT}")
print("=" * 60)

combined.write \
    .mode("overwrite") \
    .partitionBy("source") \
    .parquet(OUTPUT)

print("Feature engineering complete!")
print(f"Output written to: {OUTPUT}")

spark.stop()
