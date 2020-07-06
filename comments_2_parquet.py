#!/usr/bin/env python3

import pyspark
from pyspark.sql import functions as f
from pyspark.sql.types import *
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext

conf = SparkConf().setAppName("Reddit comments to parquet")
conf = conf.set('spark.sql.crossJoin.enabled',"true")

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

globstr = "/gscratch/comdata/raw_data/reddit_dumps/comments/RC_20*.bz2"

import re
import glob
import json
from subprocess import Popen, PIPE
from datetime import datetime
import pandas as pd
from multiprocessing import Pool

def open_fileset(globstr):
    files = glob.glob(globstr)
    for fh in files:
        print(fh)
        lines = open_input_file(fh)
        for line in lines:
            yield json.loads(line)
                
def open_input_file(input_filename):
    if re.match(r'.*\.7z$', input_filename):
        cmd = ["7za", "x", "-so", input_filename, '*'] 
    elif re.match(r'.*\.gz$', input_filename):
        cmd = ["zcat", input_filename] 
    elif re.match(r'.*\.bz2$', input_filename):
        cmd = ["bzcat", "-dk", input_filename] 

    elif re.match(r'.*\.bz', input_filename):
        cmd = ["bzcat", "-dk", input_filename] 
    elif re.match(r'.*\.xz', input_filename):
        cmd = ["xzcat",'-dk',input_filename]
    try:
        input_file = Popen(cmd, stdout=PIPE).stdout
    except NameError:
        input_file = open(input_filename, 'r')
    return input_file

def include_row(comment, subreddits_to_track = []):
    
    subreddit = comment['subreddit'].lower()

    return subreddit in subreddits_to_track

def parse_comment(comment, names= None):
    if names is None:
        names = ["id","subreddit","link_id","parent_id","created_utc","author","ups","downs","score","edited","subreddit_type","subreddit_id","stickied","is_submitter","body","error"]

    try:
        comment = json.loads(comment)
    except json.decoder.JSONDecodeError as e:
        print(e)
        print(comment)
        row = [None for _ in names]
        row[-1] = "json.decoder.JSONDecodeError|{0}|{1}".format(e,comment)
        return tuple(row)

    row = []
    for name in names:
        if name == 'created_utc':
            row.append(datetime.fromtimestamp(int(comment['created_utc']),tz=None))
        elif name == 'edited':
            val = comment[name]
            if type(val) == bool:
                row.append(val)
                row.append(None)
            else:
                row.append(True)
                row.append(datetime.fromtimestamp(int(val),tz=None))
        elif name == "time_edited":
            continue
        elif name not in comment:
            row.append(None)

        else:
            row.append(comment[name])

    return tuple(row)


#    conf = sc._conf.setAll([('spark.executor.memory', '20g'), ('spark.app.name', 'extract_reddit_timeline'), ('spark.executor.cores', '26'), ('spark.cores.max', '26'), ('spark.driver.memory','84g'),('spark.driver.maxResultSize','0'),('spark.local.dir','/gscratch/comdata/spark_tmp')])
    
sqlContext = pyspark.SQLContext(sc)

comments = sc.textFile(globstr)

schema = StructType().add("id", StringType(), True)
schema = schema.add("subreddit", StringType(), True)
schema = schema.add("link_id", StringType(), True)
schema = schema.add("parent_id", StringType(), True)
schema = schema.add("created_utc", TimestampType(), True)
schema = schema.add("author", StringType(), True)
schema = schema.add("ups", LongType(), True)
schema = schema.add("downs", LongType(), True)
schema = schema.add("score", LongType(), True)
schema = schema.add("edited", BooleanType(), True)
schema = schema.add("time_edited", TimestampType(), True)
schema = schema.add("subreddit_type", StringType(), True)
schema = schema.add("subreddit_id", StringType(), True)
schema = schema.add("stickied", BooleanType(), True)
schema = schema.add("is_submitter", BooleanType(), True)
schema = schema.add("body", StringType(), True)
schema = schema.add("error", StringType(), True)

rows = comments.map(lambda c: parse_comment(c, schema.fieldNames()))
#!/usr/bin/env python3

df =  sqlContext.createDataFrame(rows, schema)

df = df.withColumn("subreddit_2", f.lower(f.col('subreddit')))
df = df.drop('subreddit')
df = df.withColumnRenamed('subreddit_2','subreddit')

df = df.withColumnRenamed("created_utc","CreatedAt")
df = df.withColumn("Month",f.month(f.col("CreatedAt")))
df = df.withColumn("Year",f.year(f.col("CreatedAt")))
df = df.withColumn("Day",f.dayofmonth(f.col("CreatedAt")))
df = df.withColumn("subreddit_hash",f.sha2(f.col("subreddit"), 256)[0:3])
df2 = df.sort(["subreddit","author","link_id","parent_id","Year","Month","Day"],ascending=True)
df2.write.parquet("/gscratch/comdata/output/reddit_comments_by_subreddit.parquet", partitionBy=["Year",'Month'],mode='overwrite')

df3 = df.sort(["author","CreatetdAt","subreddit","link_id","parent_id","Year","Month","Day"],ascending=True)
df3.write.parquet("/gscratch/comdata/output/reddit_comments_by_author.parquet", partitionBy=["Year",'Month'],mode='overwrite')
