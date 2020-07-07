pimport pyarrow.dataset as ds
from itertools import chain, groupby, islice

# A pyarrow dataset abstracts reading, writing, or filtering a parquet file. It does not read dataa into memory. 
#dataset = ds.dataset(pathlib.Path('/gscratch/comdata/output/reddit_submissions_by_subreddit.parquet/'), format='parquet', partitioning='hive')
dataset = ds.dataset('/gscratch/comdata/output/reddit_submissions_by_author.parquet', format='parquet')

# let's get all the comments to two subreddits:
subreddits_to_pull = ['seattlewa','seattle']

# instead of loading the data into a pandas dataframe all at once we can stream it. This lets us start working with it while it is read.
scan_tasks = dataset.scan(filter = ds.field('subreddit').isin(subreddits_to_pull), columns=['id','subreddit','CreatedAt','author','ups','downs','score','subreddit_id','stickied','title','url','is_self','selftext'])

# simple function to execute scantasks and create a stream of rows 
def iterate_rows(scan_tasks):
    for st in scan_tasks:
        for rb in st.execute():
            df = rb.to_pandas()
            for t in df.itertuples():
                yield t

row_iter = iterate_rows(scan_tasks)

# now we can use python's groupby function to read one author at a time
# note that the same author can appear more than once since the record batches may not be in the correct order.
author_submissions = groupby(row_iter, lambda row: row.author)

count_dict = {}

for auth, posts in author_submissions:
    if auth in count_dict:
        count_dict[auth] = count_dict[auth] + 1
    else:
        count_dict[auth] = 1

# since it's partitioned and sorted by author, we get one group for each author 
any([ v != 1 for k,v in count_dict.items()])

