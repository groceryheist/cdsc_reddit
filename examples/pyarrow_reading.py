import pyarrow.dataset as ds

# A pyarrow dataset abstracts reading, writing, or filtering a parquet file. It does not read dataa into memory. 
#dataset = ds.dataset(pathlib.Path('/gscratch/comdata/output/reddit_submissions_by_subreddit.parquet/'), format='parquet', partitioning='hive')
dataset = ds.dataset('/gscratch/comdata/output/reddit_comments_by_subreddit.parquet/', format='parquet')

# let's get all the comments to two subreddits:
subreddits_to_pull = ['seattle','seattlewa']

# a table is a low-level structured data format.  This line pulls data into memory. Setting metadata_n_threads > 1 gives a little speed boost.
table = dataset.to_table(filter = ds.field('subreddit').isin(subreddits_to_pull), columns=['id','subreddit','CreatedAt','author','ups','downs','score','subreddit_id','stickied','title','url','is_self','selftext'])

# Since data from just these 2 subreddits fits in memory we can just turn our table into a pandas dataframe.
df = table.to_pandas()

# We should save this smaller dataset so we don't have to wait 15 min to pull from parquet next time.
df.to_csv("mydataset.csv")
