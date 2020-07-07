#!/usr/bin/env python3

# spark script to make sorted, and partitioned parquet files 

import pyspark
from pyspark.sql import functions as f
from pyspark.sql.types import *
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext
import os

spark = SparkSession.builder.getOrCreate()

sc = spark.sparkContext

conf = SparkConf().setAppName("Reddit submissions to parquet")
conf = conf.set("spark.sql.shuffle.partitions",2000)
conf = conf.set('spark.sql.crossJoin.enabled',"true")
conf = conf.set('spark.debug.maxToStringFields',200)
sqlContext = pyspark.SQLContext(sc)

df = spark.read.parquet("/gscratch/comdata/output/reddit_submissions.parquet_temp")

df = df.withColumn("subreddit_2", f.lower(f.col('subreddit')))
df = df.drop('subreddit')
df = df.withColumnRenamed('subreddit_2','subreddit')
df = df.withColumnRenamed("created_utc","CreatedAt")
df = df.withColumn("Month",f.month(f.col("CreatedAt")))
df = df.withColumn("Year",f.year(f.col("CreatedAt")))
df = df.withColumn("Day",f.dayofmonth(f.col("CreatedAt")))
df = df.withColumn("subreddit_hash",f.sha2(f.col("subreddit"), 256)[0:3])

# next we gotta resort it all.
df2 = df.sort(["subreddit","author","id","Year","Month","Day"],ascending=True)
df2.write.parquet("/gscratch/comdata/output/reddit_submissions_by_subreddit.parquet", partitionBy=["Year",'Month'], mode='overwrite')


# # we also want to have parquet files sorted by author then reddit. 
df3 = df.sort(["author","CreatedAt","subreddit","id","Year","Month","Day"],ascending=True)
df3.write.parquet("/gscratch/comdata/output/reddit_submissions_by_author.parquet", partitionBy=["Year",'Month'], mode='overwrite')

os.remove("/gscratch/comdata/output/reddit_submissions.parquet_temp")
