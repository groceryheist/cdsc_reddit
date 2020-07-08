#!/usr/bin/env python3

# spark script to make sorted, and partitioned parquet files 

import pyspark
from pyspark.sql import functions as f
from pyspark.sql import SparkSession
import os

spark = SparkSession.builder.getOrCreate()

sc = spark.sparkContext

conf = pyspark.SparkConf().setAppName("Reddit submissions to parquet")
conf = conf.set("spark.sql.shuffle.partitions",2000)
conf = conf.set('spark.sql.crossJoin.enabled',"true")
conf = conf.set('spark.debug.maxToStringFields',200)
sqlContext = pyspark.SQLContext(sc)

df = spark.read.parquet("/gscratch/comdata/output/reddit_submissions_by_subreddit.parquet")

df = df.withColumn("subreddit_2", f.lower(f.col('subreddit')))
df = df.drop('subreddit')
df = df.withColumnRenamed('subreddit_2','subreddit')
df = df.withColumnRenamed("created_utc","CreatedAt")
df = df.withColumn("Month",f.month(f.col("CreatedAt")))
df = df.withColumn("Year",f.year(f.col("CreatedAt")))
df = df.withColumn("Day",f.dayofmonth(f.col("CreatedAt")))
df = df.withColumn("subreddit_hash",f.sha2(f.col("subreddit"), 256)[0:3])

# next we gotta resort it all.
df = df.repartition("subreddit")
df2 = df.sort(["subreddit","CreatedAt","id"],ascending=True)
df2 = df.sortWithinPartitions(["subreddit","CreatedAt","id"],ascending=True)
df2.write.parquet("/gscratch/comdata/output/reddit_submissions_by_subreddit.parquet2", mode='overwrite',compression='snappy')


# # we also want to have parquet files sorted by author then reddit. 
df = df.repartition("author")
df3 = df.sort(["author","CreatedAt","id"],ascending=True)
df3 = df.sortWithinPartitions(["author","CreatedAt","id"],ascending=True)
df3.write.parquet("/gscratch/comdata/output/reddit_submissions_by_author.parquet2", mode='overwrite',compression='snappy')

os.remove("/gscratch/comdata/output/reddit_submissions.parquet_temp")
