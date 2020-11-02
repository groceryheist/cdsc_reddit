#!/usr/bin/env python3

from pyspark.sql import functions as f
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
df = spark.read.parquet("/gscratch/comdata/users/nathante/reddit_tfidf_test.parquet_temp/")

df = df.repartition(2000,'term')
df = df.sort(['term','week','subreddit'])
df = df.sortWithinPartitions(['term','week','subreddit'])

df.write.parquet("/gscratch/comdata/users/nathante/reddit_tfidf_test_sorted_tf.parquet_temp",mode='overwrite',compression='snappy')
