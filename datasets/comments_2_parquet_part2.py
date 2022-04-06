#!/usr/bin/env python3

# spark script to make sorted, and partitioned parquet files 

import pyspark
from pyspark.sql import functions as f
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

conf = pyspark.SparkConf().setAppName("Reddit submissions to parquet")
conf = conf.set("spark.sql.shuffle.partitions",2000)
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

df = df.repartition('subreddit')
df2 = df.sort(["subreddit","CreatedAt","link_id","parent_id","Year","Month","Day"],ascending=True)
df2 = df2.sortWithinPartitions(["subreddit","CreatedAt","link_id","parent_id","Year","Month","Day"],ascending=True)
df2.write.parquet("/gscratch/scrubbed/comdata/output/reddit_comments_by_subreddit.parquet", mode='overwrite', compression='snappy')

df = df.repartition('author')
df3 = df.sort(["author","CreatedAt","subreddit","link_id","parent_id","Year","Month","Day"],ascending=True)
df3 = df3.sortWithinPartitions(["author","CreatedAt","subreddit","link_id","parent_id","Year","Month","Day"],ascending=True)
df3.write.parquet("/gscratch/scrubbed/comdata/output/reddit_comments_by_author.parquet", mode='overwrite',compression='snappy')
