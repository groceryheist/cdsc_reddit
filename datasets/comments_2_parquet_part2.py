#!/usr/bin/env python3

# spark script to make sorted, and partitioned parquet files 

from pyspark.sql import functions as f
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

df = spark.read.parquet("/gscratch/comdata/output/reddit_comments.parquet_temp2",compression='snappy')

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
df2.write.parquet("/gscratch/comdata/users/nathante/reddit_comments_by_subreddit.parquet_new", mode='overwrite', compression='snappy')

df = df.repartition('author')
df3 = df.sort(["author","CreatedAt","subreddit","link_id","parent_id","Year","Month","Day"],ascending=True)
df3 = df3.sortWithinPartitions(["author","CreatedAt","subreddit","link_id","parent_id","Year","Month","Day"],ascending=True)
df3.write.parquet("/gscratch/comdata/users/nathante/reddit_comments_by_author.parquet_new", mode='overwrite',compression='snappy')
