import pyarrow.dataset as ds
from itertools import chain, groupby, islice

# A pyarrow dataset abstracts reading, writing, or filtering a parquet file. It does not read dataa into memory. 
#dataset = ds.dataset(pathlib.Path('/gscratch/comdata/output/reddit_submissions_by_subreddit.parquet/'), format='parquet', partitioning='hive')
dataset = ds.dataset('/gscratch/comdata/output/reddit_submissions_by_author.parquet', format='parquet', partitioning='hive')

# let's get all the comments to two subreddits:
subreddits_to_pull = ['seattlewa','seattle']

# instead of loading the data into a pandas dataframe all at once we can stream it. This lets us start working with it while it is read.
scan_tasks = dataset.scan(filter = ds.field('subreddit').isin(subreddits_to_pull), columns=['id','subreddit','CreatedAt','author','ups','downs','score','subreddit_id','stickied','title','url','is_self','selftext'])

# simple function to execute scantasks and create a stream of pydict rows 
def execute_scan_task(st):
    # an executed scan task yields an iterator of record_batches
    def unroll_record_batch(rb):
        df = rb.to_pandas()
        return df.itertuples()

    for rb in st.execute():
        yield unroll_record_batch(rb)


# now we just need to flatten and we have our iterator
row_iter = chain.from_iterable(chain.from_iterable(map(lambda st: execute_scan_task(st), scan_tasks)))

# now we can use python's groupby function to read one author at a time
# note that the same author can appear more than once since the record batches may not be in the correct order.
author_submissions = groupby(row_iter, lambda row: row.author)
for auth, posts in author_submissions:
    print(f"{auth} has {len(list(posts))} posts")
