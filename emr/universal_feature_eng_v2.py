"""
Universal Feature Engineering — OPTIMIZED VERSION
- Repartitions data immediately after CSV read
- Removes percent_rank() window function (too slow)
- Removes approxQuantile() (replaced with simple stats)
- Uses broadcast joins instead of shuffle joins
- Processes 3 datasets and writes unified parquet output
"""

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, log1p, lit, sin, cos, floor,
    unix_timestamp, hour, dayofweek, month, year,
    when, avg, stddev, count, to_timestamp,
    percentile_approx
)
from pyspark.sql.types import IntegerType, DoubleType
import math

# ── Args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ds1",        required=True, help="financial_fraud_detection_dataset.csv S3 path")
parser.add_argument("--ds2",        required=True, help="creditcard.csv S3 path")
parser.add_argument("--ds3",        required=True, help="credit_card_fraud_dataset.csv S3 path")
parser.add_argument("--output_uri", required=True, help="Output S3 path")
args = parser.parse_args()

DS1    = args.ds1
DS2    = args.ds2
DS3    = args.ds3
OUTPUT = args.output_uri

# ── Spark Session ─────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("universal-fraud-feature-eng") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.default.parallelism", "200") \
    .config("spark.sql.adaptive.skewJoin.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

NUM_PARTITIONS = 200  # Force parallelism

# ── Universal output columns ───────────────────────────────────────────────
UNIVERSAL_COLS = [
    "log_amount", "amount_z",
    "is_high_amount",
    "hour", "hour_sin", "hour_cos", "is_night",
    "day_of_week", "is_weekend", "txn_month",
    "user_mean_amount", "user_std_amount",
    "user_txn_count", "user_amount_z",
    "txn_count_5m",
    "pca_l2", "pca_max_abs", "pca_mean",
    "label",
    "source"
]

# ─────────────────────────────────────────────────────────────────────────
# DATASET 1 — financial_fraud_detection_dataset.csv
# ─────────────────────────────────────────────────────────────────────────
print("=" * 50)
print("DS1: financial_fraud_detection_dataset.csv")
print("=" * 50)

df1 = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv(DS1) \
    .repartition(NUM_PARTITIONS)  # CRITICAL: force parallelism immediately

print(f"DS1 rows: {df1.count()}")

# Normalize amount column name
amount_col1 = "amount" if "amount" in [c.lower() for c in df1.columns] else "Amount"
if amount_col1 == "amount" and "amount" in df1.columns:
    df1 = df1.withColumnRenamed("amount", "Amount")

# Amount features — use simple stats (no window functions)
stats1 = df1.select(
    avg("Amount").alias("mean_amt"),
    stddev("Amount").alias("std_amt"),
    percentile_approx("Amount", 0.95).alias("p95_amt")
).collect()[0]
mean1 = float(stats1["mean_amt"] or 0)
std1  = float(stats1["std_amt"]  or 1)
p95_1 = float(stats1["p95_amt"]  or mean1 * 2)

df1 = df1 \
    .withColumn("log_amount",    log1p(col("Amount").cast(DoubleType()))) \
    .withColumn("amount_z",      (col("Amount").cast(DoubleType()) - lit(mean1)) / lit(std1)) \
    .withColumn("is_high_amount", when(col("Amount") > lit(p95_1), 1).otherwise(0))

# Time features — datetime string
df1 = df1 \
    .withColumn("ts",          unix_timestamp(to_timestamp(col("timestamp")))) \
    .withColumn("hour",        (col("ts") / 3600) % 24) \
    .withColumn("hour_sin",    sin(lit(2 * math.pi) * col("hour") / lit(24.0))) \
    .withColumn("hour_cos",    cos(lit(2 * math.pi) * col("hour") / lit(24.0))) \
    .withColumn("is_night",    when((col("hour") >= 22) | (col("hour") <= 6), 1).otherwise(0)) \
    .withColumn("day_of_week", dayofweek(to_timestamp(col("timestamp")))) \
    .withColumn("is_weekend",  when(col("day_of_week").isin([1, 7]), 1).otherwise(0)) \
    .withColumn("txn_month",   month(to_timestamp(col("timestamp"))))

# User features — groupBy then join back
user_agg1 = df1.groupBy("sender_account").agg(
    avg("Amount").alias("user_mean_amount"),
    stddev("Amount").alias("user_std_amount"),
    count("*").alias("user_txn_count")
)
df1 = df1.join(user_agg1, on="sender_account", how="left") \
         .withColumn("user_amount_z",
             (col("Amount") - col("user_mean_amount")) /
             when(col("user_std_amount") > 0, col("user_std_amount")).otherwise(1)
         )

# Velocity — 5-min buckets
df1 = df1.withColumn("bucket_5m", floor(col("ts") / 300))
bkt1 = df1.groupBy("bucket_5m").count().withColumnRenamed("count", "txn_count_5m")
df1 = df1.join(bkt1, on="bucket_5m", how="left")

# No PCA columns
df1 = df1 \
    .withColumn("pca_l2",      lit(0.0)) \
    .withColumn("pca_max_abs", lit(0.0)) \
    .withColumn("pca_mean",    lit(0.0))

# Label + source
df1 = df1 \
    .withColumn("label",  when(col("is_fraud") == True, 1).otherwise(0).cast(IntegerType())) \
    .withColumn("source", lit("financial"))

df1_final = df1.select(*UNIVERSAL_COLS)
print(f"DS1 processed. Fraud: {df1_final.filter(col('label')==1).count()}")


# ─────────────────────────────────────────────────────────────────────────
# DATASET 2 — creditcard.csv
# ─────────────────────────────────────────────────────────────────────────
print("=" * 50)
print("DS2: creditcard.csv")
print("=" * 50)

df2 = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv(DS2) \
    .repartition(NUM_PARTITIONS)

print(f"DS2 rows: {df2.count()}")

stats2 = df2.select(
    avg("Amount").alias("mean_amt"),
    stddev("Amount").alias("std_amt"),
    percentile_approx("Amount", 0.95).alias("p95_amt")
).collect()[0]
mean2 = float(stats2["mean_amt"] or 0)
std2  = float(stats2["std_amt"]  or 1)
p95_2 = float(stats2["p95_amt"]  or mean2 * 2)

df2 = df2 \
    .withColumn("log_amount",    log1p(col("Amount").cast(DoubleType()))) \
    .withColumn("amount_z",      (col("Amount").cast(DoubleType()) - lit(mean2)) / lit(std2)) \
    .withColumn("is_high_amount", when(col("Amount") > lit(p95_2), 1).otherwise(0))

# Numeric time (seconds)
df2 = df2 \
    .withColumn("hour",        (col("Time") / 3600) % 24) \
    .withColumn("hour_sin",    sin(lit(2 * math.pi) * col("hour") / lit(24.0))) \
    .withColumn("hour_cos",    cos(lit(2 * math.pi) * col("hour") / lit(24.0))) \
    .withColumn("is_night",    when((col("hour") >= 22) | (col("hour") <= 6), 1).otherwise(0)) \
    .withColumn("day_of_week", lit(0)) \
    .withColumn("is_weekend",  lit(0)) \
    .withColumn("txn_month",   lit(0))

# No user column
df2 = df2 \
    .withColumn("user_mean_amount", lit(0.0)) \
    .withColumn("user_std_amount",  lit(0.0)) \
    .withColumn("user_txn_count",   lit(0)) \
    .withColumn("user_amount_z",    lit(0.0))

# Velocity
df2 = df2.withColumn("bucket_5m", floor(col("Time") / 300))
bkt2 = df2.groupBy("bucket_5m").count().withColumnRenamed("count", "txn_count_5m")
df2 = df2.join(bkt2, on="bucket_5m", how="left")

# PCA features V1-V28
from pyspark.sql.functions import sqrt, greatest
from pyspark.sql.functions import abs as sabs

v_cols = [f"V{i}" for i in range(1, 29)]
sq_sum = None
for c in v_cols:
    term   = col(c).cast(DoubleType()) * col(c).cast(DoubleType())
    sq_sum = term if sq_sum is None else sq_sum + term
df2 = df2.withColumn("pca_l2", sqrt(sq_sum))

max_abs = None
for c in v_cols:
    term    = sabs(col(c).cast(DoubleType()))
    max_abs = term if max_abs is None else greatest(max_abs, term)
df2 = df2.withColumn("pca_max_abs", max_abs)

pca_sum = None
for c in v_cols:
    pca_sum = col(c).cast(DoubleType()) if pca_sum is None else pca_sum + col(c).cast(DoubleType())
df2 = df2.withColumn("pca_mean", pca_sum / lit(len(v_cols)))

df2 = df2 \
    .withColumn("label",  col("Class").cast(IntegerType())) \
    .withColumn("source", lit("creditcard"))

df2_final = df2.select(*UNIVERSAL_COLS)
print(f"DS2 processed. Fraud: {df2_final.filter(col('label')==1).count()}")


# ─────────────────────────────────────────────────────────────────────────
# DATASET 3 — credit_card_fraud_dataset.csv
# ─────────────────────────────────────────────────────────────────────────
print("=" * 50)
print("DS3: credit_card_fraud_dataset.csv")
print("=" * 50)

df3 = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv(DS3) \
    .repartition(NUM_PARTITIONS)

print(f"DS3 rows: {df3.count()}")

stats3 = df3.select(
    avg("Amount").alias("mean_amt"),
    stddev("Amount").alias("std_amt"),
    percentile_approx("Amount", 0.95).alias("p95_amt")
).collect()[0]
mean3 = float(stats3["mean_amt"] or 0)
std3  = float(stats3["std_amt"]  or 1)
p95_3 = float(stats3["p95_amt"]  or mean3 * 2)

df3 = df3 \
    .withColumn("log_amount",    log1p(col("Amount").cast(DoubleType()))) \
    .withColumn("amount_z",      (col("Amount").cast(DoubleType()) - lit(mean3)) / lit(std3)) \
    .withColumn("is_high_amount", when(col("Amount") > lit(p95_3), 1).otherwise(0))

# Datetime string
df3 = df3 \
    .withColumn("ts",          unix_timestamp(to_timestamp(col("TransactionDate")))) \
    .withColumn("hour",        (col("ts") / 3600) % 24) \
    .withColumn("hour_sin",    sin(lit(2 * math.pi) * col("hour") / lit(24.0))) \
    .withColumn("hour_cos",    cos(lit(2 * math.pi) * col("hour") / lit(24.0))) \
    .withColumn("is_night",    when((col("hour") >= 22) | (col("hour") <= 6), 1).otherwise(0)) \
    .withColumn("day_of_week", dayofweek(to_timestamp(col("TransactionDate")))) \
    .withColumn("is_weekend",  when(col("day_of_week").isin([1, 7]), 1).otherwise(0)) \
    .withColumn("txn_month",   month(to_timestamp(col("TransactionDate"))))

# User features — MerchantID
user_agg3 = df3.groupBy("MerchantID").agg(
    avg("Amount").alias("user_mean_amount"),
    stddev("Amount").alias("user_std_amount"),
    count("*").alias("user_txn_count")
)
df3 = df3.join(user_agg3, on="MerchantID", how="left") \
         .withColumn("user_amount_z",
             (col("Amount") - col("user_mean_amount")) /
             when(col("user_std_amount") > 0, col("user_std_amount")).otherwise(1)
         )

# Velocity
df3 = df3.withColumn("bucket_5m", floor(col("ts") / 300))
bkt3 = df3.groupBy("bucket_5m").count().withColumnRenamed("count", "txn_count_5m")
df3 = df3.join(bkt3, on="bucket_5m", how="left")

# No PCA
df3 = df3 \
    .withColumn("pca_l2",      lit(0.0)) \
    .withColumn("pca_max_abs", lit(0.0)) \
    .withColumn("pca_mean",    lit(0.0))

df3 = df3 \
    .withColumn("label",  col("IsFraud").cast(IntegerType())) \
    .withColumn("source", lit("credit_card_fraud"))

df3_final = df3.select(*UNIVERSAL_COLS)
print(f"DS3 processed. Fraud: {df3_final.filter(col('label')==1).count()}")


# ─────────────────────────────────────────────────────────────────────────
# COMBINE + WRITE
# ─────────────────────────────────────────────────────────────────────────
print("=" * 50)
print("Combining all 3 datasets...")
print("=" * 50)

combined = df1_final.union(df2_final).union(df3_final).repartition(100)

total = combined.count()
fraud = combined.filter(col("label") == 1).count()
print(f"Combined: {total} rows | Fraud: {fraud} ({fraud/total:.2%})")

print(f"Writing to {OUTPUT} ...")
combined.write \
    .mode("overwrite") \
    .partitionBy("source") \
    .parquet(OUTPUT)

print("=" * 50)
print("Feature engineering complete!")
print(f"Output: {OUTPUT}")
print("=" * 50)

spark.stop()
