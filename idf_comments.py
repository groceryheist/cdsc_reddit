from pyspark.sql import functions as f
from pyspark.sql import SparkSession
from pyspark.sql import Window

## TODO:need to exclude automoderator / bot posts.
## TODO:need to exclude better handle hyperlinks. 

spark = SparkSession.builder.getOrCreate()
df = spark.read.parquet("/gscratch/comdata/users/nathante/reddit_tfidf_test.parquet_temp")

include_subs = set(open("/gscratch/comdata/users/nathante/cdsc-reddit/top_25000_subs_by_comments.txt"))
include_subs = {s.strip('\n') for s in include_subs}

# aggregate counts by week. now subreddit-term is distinct
df = df.filter(df.subreddit.isin(include_subs))
df = df.groupBy(['subreddit','term']).agg(f.sum('tf').alias('tf'))

max_subreddit_terms = df.groupby(['subreddit']).max('tf') # subreddits are unique
max_subreddit_terms = max_subreddit_terms.withColumnRenamed('max(tf)','sr_max_tf')

df = df.join(max_subreddit_terms, on='subreddit')

df = df.withColumn("relative_tf", df.tf / df.sr_max_tf)

# group by term. term is unique
idf = df.groupby(['term']).count()

N_docs = df.select('subreddit').distinct().count()

idf = idf.withColumn('idf',f.log(N_docs/f.col('count')))

# collect the dictionary to make a pydict of terms to indexes
terms = idf.select('term').distinct() # terms are distinct
terms = terms.withColumn('term_id',f.row_number().over(Window.orderBy("term"))) # term ids are distinct

# make subreddit ids
subreddits = df.select(['subreddit']).distinct()
subreddits = subreddits.withColumn('subreddit_id',f.row_number().over(Window.orderBy("subreddit")))

df = df.join(subreddits,on='subreddit')

# map terms to indexes in the tfs and the idfs
df = df.join(terms,on='term') # subreddit-term-id is unique

idf = idf.join(terms,on='term')

# join on subreddit/term to create tf/dfs indexed by term
df = df.join(idf, on=['term_id','term'])

# agg terms by subreddit to make sparse tf/df vectors
df = df.withColumn("tf_idf", (0.5 + (0.5 * df.relative_tf) * df.idf))

df.write.parquet('/gscratch/comdata/users/nathante/subreddit_tfidf.parquet',mode='overwrite',compression='snappy')
