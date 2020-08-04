import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from itertools import groupby, islice, chain
import fire
from collections import Counter
import pandas as pd
import os
import datetime
from nltk import wordpunct_tokenize, MWETokenizer

# compute term frequencies for comments in each subreddit by week
def weekly_tf(partition):
    dataset = ds.dataset(f'/gscratch/comdata/output/reddit_comments_by_subreddit.parquet/{partition}', format='parquet')
    batches = dataset.to_batches(columns=['CreatedAt','subreddit','body'])

    schema = pa.schema([pa.field('subreddit', pa.string(), nullable=False),
                        pa.field('term', pa.string(), nullable=False),
                        pa.field('week', pa.date32(), nullable=False),
                        pa.field('tf', pa.int64(), nullable=False)]
    )

    dfs = (b.to_pandas() for b in batches)

    def add_week(df):
        df['week'] = (df.CreatedAt - pd.to_timedelta(df.CreatedAt.dt.dayofweek, unit='d')).dt.date
        return(df)

    dfs = (add_week(df) for df in dfs)

    def iterate_rows(dfs):
        for df in dfs:
            for row in df.itertuples():
                yield row

    rows = iterate_rows(dfs)

    subreddit_weeks = groupby(rows, lambda r: (r.subreddit, r.week))

    tokenizer = MWETokenizer()

    def tf_comments(subreddit_weeks):
        for key, posts in subreddit_weeks:
            subreddit, week = key
            tfs = Counter([])

            for post in posts:
                tfs.update(tokenizer.tokenize(wordpunct_tokenize(post.body.lower())))

            for term, tf in tfs.items():
                yield [subreddit, term, week, tf]
            
    outrows = tf_comments(subreddit_weeks)

    outchunksize = 10000

    with pq.ParquetWriter("/gscratch/comdata/users/nathante/reddit_tfidf_test.parquet_temp/{partition}",schema=schema,compression='snappy',flavor='spark') as writer:
        while True:
            chunk = islice(outrows,outchunksize)
            pddf = pd.DataFrame(chunk, columns=schema.names)
            print(pddf)
            table = pa.Table.from_pandas(pddf,schema=schema)
            if table.shape[0] == 0:
                break
            writer.write_table(table)

        writer.close()


def gen_task_list():
    files = os.listdir("/gscratch/comdata/output/reddit_comments_by_subreddit.parquet/")
    with open("tf_task_list",'w') as outfile:
        for f in files:
            if f.endswith(".parquet"):
                outfile.write(f"python3 tf_reddit_comments.py weekly_tf {f}\n")

if __name__ == "__main__":
    fire.Fire({"gen_task_list":gen_task_list,
               "weekly_tf":weekly_tf})
