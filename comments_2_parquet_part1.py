#!/usr/bin/env python3
import json
from datetime import datetime
from multiprocessing import Pool
from itertools import islice
from helper import find_dumps, open_fileset
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

globstr_base = "/gscratch/comdata/reddit_dumps/comments/RC_20*"

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

dumpdir = "/gscratch/comdata/raw_data/reddit_dumps/comments"

files = list(find_dumps(dumpdir, base_pattern="RC_20*.*"))

pool = Pool(28)

stream = open_fileset(files)

N = 100000

rows = pool.imap_unordered(parse_comment, stream, chunksize=int(N/28))

schema = pa.schema([
    pa.field('id', pa.string(), nullable=True),
    pa.field('subreddit', pa.string(), nullable=True),
    pa.field('link_id', pa.string(), nullable=True),
    pa.field('parent_id', pa.string(), nullable=True),
    pa.field('created_utc', pa.timestamp('ms'), nullable=True),
    pa.field('author', pa.string(), nullable=True),
    pa.field('ups', pa.int64(), nullable=True),
    pa.field('downs', pa.int64(), nullable=True),
    pa.field('score', pa.int64(), nullable=True),
    pa.field('edited', pa.bool_(), nullable=True),
    pa.field('time_edited', pa.timestamp('ms'), nullable=True),
    pa.field('subreddit_type', pa.string(), nullable=True),
    pa.field('subreddit_id', pa.string(), nullable=True),
    pa.field('stickied', pa.bool_(), nullable=True),
    pa.field('is_submitter', pa.bool_(), nullable=True),
    pa.field('body', pa.string(), nullable=True),
    pa.field('error', pa.string(), nullable=True),
])

with pq.ParquetWriter("/gscratch/comdata/output/reddit_comments.parquet_temp",schema=schema,compression='snappy',flavor='spark') as writer:
    while True:
        chunk = islice(rows,N)
        pddf = pd.DataFrame(chunk, columns=schema.names)
        table = pa.Table.from_pandas(pddf,schema=schema)
        if table.shape[0] == 0:
            break
        writer.write_table(table)

    writer.close()
