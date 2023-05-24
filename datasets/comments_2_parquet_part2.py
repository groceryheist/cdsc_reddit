#!/usr/bin/env python3

# spark script to make sorted, and partitioned parquet files 

import pyspark
from pyspark.sql import functions as f
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

conf = pyspark.SparkConf().setAppName("Reddit submissions to parquet")
conf = conf.set("spark.sql.shuffle.partitions",2400)
conf = conf.set('spark.sql.crossJoin.enabled',"true")
conf = conf.set('spark.debug.maxToStringFields',200)
sc = spark.sparkContext

df = spark.read.parquet("/gscratch/comdata/output/temp/reddit_comments.parquet",compression='snappy')

df = df.withColumn("subreddit_2", f.lower(f.col('subreddit')))
df = df.drop('subreddit')
df = df.withColumnRenamed('subreddit_2','subreddit')

df = df.withColumnRenamed("created_utc","CreatedAt")
df = df.withColumn("Month",f.month(f.col("CreatedAt")))
df = df.withColumn("Year",f.year(f.col("CreatedAt")))
df = df.withColumn("Day",f.dayofmonth(f.col("CreatedAt")))

df = df.repartition(2400,'author','subreddit',"Year","Month","Day")
df3 = df.sort(["author","subreddit","Year","Month","Day","CreatedAt","link_id","parent_id"],ascending=True)
df3 = df3.sortWithinPartitions(["author","subreddit","Year","Month","Day","CreatedAt","link_id","parent_id"],ascending=True)
df3.write.parquet("/gscratch/scrubbed/comdata/reddit_comments_by_author.parquet", mode='overwrite',compression='snappy')
