from pyspark.sql import functions as f
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
df = spark.read.parquet("/gscratch/comdata/users/nathante/reddit_tfidf_test_authors.parquet_temp/")

max_subreddit_week_authors = df.groupby(['subreddit','week']).max('tf')
max_subreddit_week_authors = max_subreddit_week_authors.withColumnRenamed('max(tf)','sr_week_max_tf')

df = df.join(max_subreddit_week_authors, ['subreddit','week'])

df = df.withColumn("relative_tf", df.tf / df.sr_week_max_tf)

# group by term / week
idf = df.groupby(['author','week']).count()

idf = idf.withColumnRenamed('count','idf')

# output: term | week | df
#idf.write.parquet("/gscratch/comdata/users/nathante/reddit_tfidf_test_sorted_tf.parquet_temp",mode='overwrite',compression='snappy')

# collect the dictionary to make a pydict of terms to indexes
authors = idf.select('author').distinct()
authors = authors.withColumn('author_id',f.monotonically_increasing_id())


# map terms to indexes in the tfs and the idfs
df = df.join(terms,on='author')

idf = idf.join(terms,on='author')

# join on subreddit/term/week to create tf/dfs indexed by term
df = df.join(idf, on=['author_id','week','author'])

# agg terms by subreddit to make sparse tf/df vectors
df = df.withColumn("tf_idf",df.relative_tf / df.sr_week_max_tf)

df = df.groupby(['subreddit','week']).agg(f.collect_list(f.struct('term_id','tf_idf')).alias('tfidf_maps'))
 
df = df.withColumn('tfidf_vec', f.map_from_entries('tfidf_maps'))

# output: subreddit | week | tf/df
df.write.parquet('/gscratch/comdata/users/nathante/test_tfidf_authors.parquet',mode='overwrite',compression='snappy')
