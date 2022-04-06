#!/usr/bin/env python3

# two stages:
# 1. from gz to arrow parquet (this script) 
# 2. from arrow parquet to spark parquet (submissions_2_parquet_part2.py)
from datetime import datetime
from pathlib import Path
from itertools import islice
from helper import find_dumps, open_fileset
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import fire
import os
import json

def parse_submission(post, names = None):
    if names is None:
        names = ['id','author','subreddit','title','created_utc','permalink','url','domain','score','ups','downs','over_18','has_media','selftext','retrieved_on','num_comments','gilded','edited','time_edited','subreddit_type','subreddit_id','subreddit_subscribers','name','is_self','stickied','quarantine','error']

    try:
        post = json.loads(post)
    except (ValueError) as e:
        #        print(e)
        #        print(post)
        row = [None for _ in names]
        row[-1] = "Error parsing json|{0}|{1}".format(e,post)
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

def parse_dump(partition):

    N=10000
    stream = open_fileset([f"/gscratch/comdata/raw_data/reddit_dumps/submissions/{partition}"])
    rows = map(parse_submission,stream)
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
        pa.field('quarantine',pa.bool_(),nullable=True),
        pa.field('error',pa.string(),nullable=True)])

    Path("/gscratch/comdata/output/temp/reddit_submissions.parquet/").mkdir(exist_ok=True,parents=True)

    with pq.ParquetWriter(f"/gscratch/comdata/output/temp/reddit_submissions.parquet/{partition}",schema=schema,compression='snappy',flavor='spark') as writer:
        while True:
            chunk = islice(rows,N)
            pddf = pd.DataFrame(chunk, columns=schema.names)
            table = pa.Table.from_pandas(pddf,schema=schema)
            if table.shape[0] == 0:
                break
            writer.write_table(table)

        writer.close()

def gen_task_list(dumpdir="/gscratch/comdata/raw_data/reddit_dumps/submissions"):
    files = list(find_dumps(dumpdir,base_pattern="RS_20*.*"))
    with open("submissions_task_list.sh",'w') as of:
        for fpath in files:
            partition = os.path.split(fpath)[1]
            of.write(f'python3 submissions_2_parquet_part1.py parse_dump {partition}\n')

if __name__ == "__main__":
    fire.Fire({'parse_dump':parse_dump,
              'gen_task_list':gen_task_list})
