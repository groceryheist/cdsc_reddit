#!/usr/bin/env python3

# two stages:
# 1. from gz to arrow parquet
# 2. from arrow parquet to spark parquet

from collections import defaultdict
from os import path
import glob
import json
import re
from datetime import datetime
from subprocess import Popen, PIPE
from multiprocessing import Pool, SimpleQueue

dumpdir = "/gscratch/comdata/raw_data/reddit_dumps/submissions"

def find_json_files(dumpdir):
    base_pattern = "RS_20*.*"

    files = glob.glob(path.join(dumpdir,base_pattern))

    # build a dictionary of possible extensions for each dump
    dumpext = defaultdict(list)
    for fpath in files:
        fname, ext = path.splitext(fpath)
        dumpext[fname].append(ext)

    ext_priority = ['.zst','.xz','.bz2']

    for base, exts in dumpext.items():
        found = False
        if len(exts) == 1:
            yield base + exts[0]
            found = True
        else:
            for ext in ext_priority:
                if ext in exts:
                    yield base + ext
                    found = True
        assert(found == True)

files = list(find_json_files(dumpdir))

def read_file(fh):
    lines = open_input_file(fh)
    for line in lines:
        yield line

def open_fileset(files):
    for fh in files:
        print(fh)
        lines = open_input_file(fh)
        for line in lines:
            yield line

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
        cmd = ["xzcat",'-dk', '-T 20',input_filename]
    elif re.match(r'.*\.zst',input_filename):
        cmd = ['zstd','-dck', input_filename]
    try:
        input_file = Popen(cmd, stdout=PIPE).stdout
    except NameError as e:
        print(e)
        input_file = open(input_filename, 'r')
    return input_file


def parse_submission(post, names = None):
    if names is None:
        names = ['id','author','subreddit','title','created_utc','permalink','url','domain','score','ups','downs','over_18','has_media','selftext','retrieved_on','num_comments','gilded','edited','time_edited','subreddit_type','subreddit_id','subreddit_subscribers','name','is_self','stickied','is_submitter','quarantine','error']

    try:
        post = json.loads(post)
    except (json.decoder.JSONDecodeError, UnicodeDecodeError) as e:
        #        print(e)
        #        print(post)
        row = [None for _ in names]
        row[-1] = "json.decoder.JSONDecodeError|{0}|{1}".format(e,post)
        return tuple(row)

    row = []

    for name in names:
        if name == 'created_utc' or name == 'retrieved_on':
            val = post.get(name,None)
            if val is not None:
                row.append(datetime.fromtimestamp(int(post[name]),tz=None))
            else:
                row.append(None)
        elif name == 'edited':
            val = post[name]
            if type(val) == bool:
                row.append(val)
                row.append(None)
            else:
                row.append(True)
                row.append(datetime.fromtimestamp(int(val),tz=None))
        elif name == "time_edited":
            continue
        elif name == 'has_media':
            row.append(post.get('media',None) is not None)

        elif name not in post:
            row.append(None)
        else:
            row.append(post[name])
    return tuple(row)

pool = Pool(28)

stream = open_fileset(files)

N = 100000

rows = pool.imap_unordered(parse_submission, stream, chunksize=int(N/28))

from itertools import islice
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

schema = pa.schema([
    pa.field('id', pa.string(),nullable=True),
    pa.field('author', pa.string(),nullable=True),
    pa.field('subreddit', pa.string(),nullable=True),
    pa.field('title', pa.string(),nullable=True),
    pa.field('created_utc', pa.timestamp('ms'),nullable=True),
    pa.field('permalink', pa.string(),nullable=True),
    pa.field('url', pa.string(),nullable=True),
    pa.field('domain', pa.string(),nullable=True),
    pa.field('score', pa.int64(),nullable=True),
    pa.field('ups', pa.int64(),nullable=True),
    pa.field('downs', pa.int64(),nullable=True),
    pa.field('over_18', pa.bool_(),nullable=True),
    pa.field('has_media',pa.bool_(),nullable=True),
    pa.field('selftext',pa.string(),nullable=True),
    pa.field('retrieved_on', pa.timestamp('ms'),nullable=True),
    pa.field('num_comments', pa.int64(),nullable=True),
    pa.field('gilded',pa.int64(),nullable=True),
    pa.field('edited',pa.bool_(),nullable=True),
    pa.field('time_edited',pa.timestamp('ms'),nullable=True),
    pa.field('subreddit_type',pa.string(),nullable=True),
    pa.field('subreddit_id',pa.string(),nullable=True),
    pa.field('subreddit_subscribers',pa.int64(),nullable=True),
    pa.field('name',pa.string(),nullable=True),
    pa.field('is_self',pa.bool_(),nullable=True),
    pa.field('stickied',pa.bool_(),nullable=True),
    pa.field('is_submitter',pa.bool_(),nullable=True),
    pa.field('quarantine',pa.bool_(),nullable=True),
    pa.field('error',pa.string(),nullable=True)])

with  pq.ParquetWriter("/gscratch/comdata/output/reddit_submissions.parquet_temp",schema=schema,compression='snappy',flavor='spark') as writer:
    while True:
        chunk = islice(rows,N)
        pddf = pd.DataFrame(chunk, columns=schema.names)
        table = pa.Table.from_pandas(pddf,schema=schema)
        if table.shape[0] == 0:
            break
        writer.write_table(table)

    writer.close()

import pyspark
from pyspark.sql import functions as f
from pyspark.sql.types import *
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

conf = SparkConf().setAppName("Reddit submissions to parquet")
conf = conf.set('spark.sql.crossJoin.enabled',"true")

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


# we also want to have parquet files sorted by author then reddit. 
df3 = df.sort(["author","CreatedAt","subreddit","id","Year","Month","Day"],ascending=True)
df3.write.parquet("/gscratch/comdata/output/reddit_submissions_by_author.parquet", partitionBy=["Year",'Month'], mode='overwrite')

os.remove("/gscratch/comdata/output/reddit_submissions.parquet_temp")
