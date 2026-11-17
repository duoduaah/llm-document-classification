import argparse
import os
import sys

import yaml
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

OUT_SCHEMA = T.StructType([
    T.StructField("doc_id", T.StringType()), T.StructField("source", T.StringType()),
    T.StructField("role", T.StringType()), T.StructField("label", T.StringType()),
    T.StructField("lang_declared", T.StringType()), T.StructField("meta", T.StringType()),
    T.StructField("text_norm", T.StringType()), T.StructField("n_tokens", T.IntegerType()),
    T.StructField("content_hash", T.StringType()), T.StructField("lang_detected", T.StringType()),
])


def process_batches(batches):
    """Runs inside Spark workers (pandas batches)."""
    import docclf
    for pdf in batches:
        text = [docclf.normalize(t, s) for t, s in zip(pdf["text"], pdf["source"])]
        out = pdf[["doc_id", "source", "role", "label", "lang_declared", "meta"]].copy()
        out["text_norm"] = text
        out["n_tokens"] = [docclf.n_tokens(t) for t in text]
        out["content_hash"] = [docclf.content_hash(t) for t in text]
        out["lang_detected"] = [docclf.detect_lang(t) for t in text]
        yield out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    pc, seed = cfg["prepare"], cfg["seed"]

    spark = (SparkSession.builder.master("local[*]").appName("docclf-prepare")
             .config("spark.driver.memory", pc["spark"]["driver_memory"])
             .config("spark.sql.shuffle.partitions", str(pc["spark"]["shuffle_partitions"]))
             .getOrCreate())
    spark.sparkContext.addPyFile(os.path.join(SRC_DIR, "docclf.py"))
    spark.sparkContext.setLogLevel("ERROR")

    raw = spark.read.parquet(cfg["paths"]["canonical_raw"])
    df = raw.mapInPandas(process_batches, schema=OUT_SCHEMA)

    # drop very short documents (per-source minimum from config)
    keep = F.lit(True)
    for src, mt in pc["min_tokens"].items():
        keep = keep & ~((F.col("source") == src) & (F.col("n_tokens") < mt))
    df = df.filter(keep).dropDuplicates(["content_hash"]).cache()

    # ---- balance + split the main (English) pool by label
    main_df = df.filter(F.col("role") == "main")
    w = Window.partitionBy("label").orderBy(F.xxhash64(F.col("doc_id"), F.lit(seed)))
    main_df = main_df.withColumn("rk", F.row_number().over(w))
    counts = main_df.groupBy("label").agg(F.count("*").alias("n_label"))
    main_df = main_df.join(counts, "label")

    sc, ff = pc["split_counts"], pc["fallback_fractions"]
    n_use = F.least(F.col("n_label"), F.lit(pc["max_per_class"]))
    fixed = F.col("n_label") >= F.lit(sc["train"] + sc["val"] + sc["test"])
    tr = F.when(fixed, F.lit(sc["train"])).otherwise(F.floor(n_use * ff["train"]))
    va = tr + F.when(fixed, F.lit(sc["val"])).otherwise(F.floor(n_use * ff["val"]))
    te = F.when(fixed, va + F.lit(sc["test"])).otherwise(n_use)
    main_df = main_df.withColumn(
        "split",
        F.when(F.col("rk") <= tr, "train").when(F.col("rk") <= va, "val")
         .when(F.col("rk") <= te, "test").when(F.col("rk") <= n_use, "spare").otherwise("unused"),
    ).drop("rk", "n_label")

    other = df.filter(F.col("role") != "main").withColumn("split", F.col("role"))
    result = main_df.unionByName(other).filter(F.col("split") != "unused")

    out = cfg["paths"]["processed"]
    result.write.mode("overwrite").parquet(out)
    print(f"wrote {out}")

    res = spark.read.parquet(out)
    print("\n== documents per label x split (main pool)")
    res.filter(F.col("role") == "main").groupBy("label", "split").count().orderBy("label", "split").show(50, False)
    print("== documents per source x language (declared)")
    res.groupBy("source", "lang_declared").count().orderBy("source", "lang_declared").show(50, False)
    print("== language-ID agreement (declared == detected) by declared language")
    (res.groupBy("lang_declared")
        .agg(F.count("*").alias("n"), F.round(F.avg((F.col("lang_declared") == F.col("lang_detected")).cast("int")), 3).alias("agree"))
        .orderBy("lang_declared").show(20, False))
    spark.stop()


if __name__ == "__main__":
    main()
