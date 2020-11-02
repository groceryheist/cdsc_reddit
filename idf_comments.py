from pyspark.sql import functions as f
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
df = spark.read.parquet("/gscratch/comdata/users/nathante/reddit_tfidf_test.parquet_temp")

max_subreddit_week_terms = df.groupby(['subreddit','week']).max('tf')
max_subreddit_week_terms = max_subreddit_week_terms.withColumnRenamed('max(tf)','sr_week_max_tf')

df = df.join(max_subreddit_week_terms, ['subreddit','week'])

df = df.withColumn("relative_tf", df.tf / df.sr_week_max_tf)

# group by term / week
idf = df.groupby(['term','week']).count()

idf = idf.withColumnRenamed('count','idf')

# output: term | week | df
#idf.write.parquet("/gscratch/comdata/users/nathante/reddit_tfidf_test_sorted_tf.parquet_temp",mode='overwrite',compression='snappy')

# collect the dictionary to make a pydict of terms to indexes
terms = idf.select('term').distinct()
terms = terms.withColumn('term_id',f.monotonically_increasing_id())


# print('collected terms')

# terms = [t.term for t in terms]
# NTerms = len(terms)
# term_id_map = {term:i for i,term in enumerate(sorted(terms))}

# term_id_map = spark.sparkContext.broadcast(term_id_map)

# print('term_id_map is broadcasted')

# def map_term(x):
#     return term_id_map.value[x]

# map_term_udf = f.udf(map_term)

# map terms to indexes in the tfs and the idfs
df = df.join(terms,on='term')

idf = idf.join(terms,on='term')

# join on subreddit/term/week to create tf/dfs indexed by term
df = df.join(idf, on=['term_id','week','term'])

# agg terms by subreddit to make sparse tf/df vectors
df = df.withColumn("tf_idf",df.relative_tf / df.sr_week_max_tf)

df = df.groupby(['subreddit','week']).agg(f.collect_list(f.struct('term_id','tf_idf')).alias('tfidf_maps'))
 
df = df.withColumn('tfidf_vec', f.map_from_entries('tfidf_maps'))

# output: subreddit | week | tf/df
df.write.parquet('/gscratch/comdata/users/nathante/test_tfidf.parquet',mode='overwrite',compression='snappy')
