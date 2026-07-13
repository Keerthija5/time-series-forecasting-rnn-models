import json
import os
import shutil
import sys


CSV_FILENAME = "airline-passengers.csv"
OUTPUT_DIR = "outputs"
PROCESSED_DIR = os.path.join("data", "processed")
FEATURE_OUTPUT = os.path.join(PROCESSED_DIR, "monthly_demand_features.csv")
SPARK_REPORT_OUTPUT = os.path.join(OUTPUT_DIR, "spark_preprocessing_report.json")


def require_pyspark():
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window
        return SparkSession, F, Window
    except ModuleNotFoundError:
        print(
            "\nPySpark is not installed in this environment.\n"
            "Install the optional Spark dependencies first:\n\n"
            "  pip install -r requirements-spark.txt\n\n"
            "Then run:\n\n"
            "  python3 spark_preprocess_timeseries.py\n"
        )
        sys.exit(1)


def write_single_csv(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_dir = output_path + "_tmp"

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    if os.path.exists(output_path):
        os.remove(output_path)

    df.coalesce(1).write.mode("overwrite").option("header", True).csv(temp_dir)
    part_files = [
        file_name for file_name in os.listdir(temp_dir)
        if file_name.startswith("part-") and file_name.endswith(".csv")
    ]
    if not part_files:
        raise RuntimeError("Spark did not create a CSV part file.")

    shutil.move(os.path.join(temp_dir, part_files[0]), output_path)
    shutil.rmtree(temp_dir)


def main():
    SparkSession, F, Window = require_pyspark()

    spark = (
        SparkSession.builder
        .appName("time-series-demand-preprocessing")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    raw_df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(CSV_FILENAME)
    )

    prepared = (
        raw_df
        .withColumn("month_date", F.to_date(F.concat(F.col("Month"), F.lit("-01"))))
        .withColumn("passengers", F.col("Passengers").cast("double"))
        .select("month_date", "passengers")
    )

    total_rows = prepared.count()
    missing_months = prepared.filter(F.col("month_date").isNull()).count()
    missing_values = prepared.filter(F.col("passengers").isNull()).count()
    duplicate_months = (
        prepared.groupBy("month_date")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )

    month_window = Window.orderBy("month_date")
    feature_df = (
        prepared
        .orderBy("month_date")
        .withColumn("lag_1", F.lag("passengers", 1).over(month_window))
        .withColumn("lag_3", F.lag("passengers", 3).over(month_window))
        .withColumn("lag_12", F.lag("passengers", 12).over(month_window))
        .withColumn(
            "rolling_mean_3",
            F.avg("passengers").over(month_window.rowsBetween(-3, -1))
        )
        .withColumn(
            "rolling_mean_12",
            F.avg("passengers").over(month_window.rowsBetween(-12, -1))
        )
        .withColumn("target_next_month", F.lead("passengers", 1).over(month_window))
        .withColumn("target_month", F.lead("month_date", 1).over(month_window))
        .filter(F.col("lag_12").isNotNull())
        .filter(F.col("target_next_month").isNotNull())
    )

    feature_count = feature_df.count()
    train_end = int(0.70 * feature_count)
    val_end = int(0.85 * feature_count)

    indexed = feature_df.withColumn("row_id", F.row_number().over(month_window))
    split_df = (
        indexed
        .withColumn(
            "split",
            F.when(F.col("row_id") <= train_end, F.lit("train"))
            .when(F.col("row_id") <= val_end, F.lit("validation"))
            .otherwise(F.lit("test"))
        )
        .drop("row_id")
    )

    split_counts = {
        row["split"]: row["count"]
        for row in split_df.groupBy("split").count().collect()
    }

    write_single_csv(split_df, FEATURE_OUTPUT)

    report = {
        "input_file": CSV_FILENAME,
        "raw_rows": total_rows,
        "feature_rows_after_lag_and_target_drop": feature_count,
        "missing_month_values": missing_months,
        "missing_passenger_values": missing_values,
        "duplicate_months": duplicate_months,
        "split_rule": "70/15/15 chronological split after lag and target creation",
        "features_created": [
            "lag_1",
            "lag_3",
            "lag_12",
            "rolling_mean_3",
            "rolling_mean_12",
            "target_next_month",
        ],
        "split_counts": split_counts,
        "processed_feature_file": FEATURE_OUTPUT,
        "note": (
            "This Spark step prepares clean forecasting features. "
            "The PyTorch script still trains the RNN models separately."
        ),
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(SPARK_REPORT_OUTPUT, "w") as f:
        json.dump(report, f, indent=2)

    print("\nSpark preprocessing completed.")
    print(f"Processed feature file: {FEATURE_OUTPUT}")
    print(f"Preprocessing report: {SPARK_REPORT_OUTPUT}")
    print(
        "\nInterpretation: Spark is used here for raw-to-clean data preparation, "
        "not because this small dataset needs big-data processing."
    )

    spark.stop()


if __name__ == "__main__":
    main()
