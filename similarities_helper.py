from pyspark.sql import Window
from pyspark.sql import functions as f
from enum import Enum
from pyspark.mllib.linalg.distributed import CoordinateMatrix

class tf_weight(Enum):
    MaxTF = 1
    Norm05 = 2


def cosine_similarities(tfidf, term_colname, min_df, included_subreddits, similarity_threshold):
    term = term_colname
    term_id = term + '_id'
    term_id_new = term + '_id_new'

    if min_df is None:
        min_df = 0.1 * len(included_subreddits)

    tfidf = tfidf.filter(f.col("subreddit").isin(included_subreddits))
    tfidf = tfidf.cache()

    # reset the subreddit ids
    sub_ids = tfidf.select('subreddit_id').distinct()
    sub_ids = sub_ids.withColumn("subreddit_id_new",f.row_number().over(Window.orderBy("subreddit_id")))
    tfidf = tfidf.join(sub_ids,'subreddit_id')

    # only use terms in at least min_df included subreddits
    new_count = tfidf.groupBy(term_id).agg(f.count(term_id).alias('new_count'))
#    new_count = new_count.filter(f.col('new_count') >= min_df)
    tfidf = tfidf.join(new_count,term_id,how='inner')
    
    # reset the term ids
    term_ids = tfidf.select([term_id]).distinct()
    term_ids = term_ids.withColumn(term_id_new,f.row_number().over(Window.orderBy(term_id)))
    tfidf = tfidf.join(term_ids,term_id)

    tfidf = tfidf.withColumnRenamed("tf_idf","tf_idf_old")
    # tfidf = tfidf.withColumnRenamed("idf","idf_old")
    # tfidf = tfidf.withColumn("idf",f.log(25000/f.col("count")))
    tfidf = tfidf.withColumn("tf_idf", tfidf.relative_tf * tfidf.idf)

    # step 1 make an rdd of entires
    # sorted by (dense) spark subreddit id
    #    entries = tfidf.filter((f.col('subreddit') == 'asoiaf') | (f.col('subreddit') == 'gameofthrones') | (f.col('subreddit') == 'christianity')).select(f.col("term_id_new")-1,f.col("subreddit_id_new")-1,"tf_idf").rdd
 
    n_partitions = int(len(included_subreddits)*2 / 5)

    entries = tfidf.select(f.col(term_id_new)-1,f.col("subreddit_id_new")-1,"tf_idf").rdd.repartition(n_partitions)

    # put like 10 subredis in each partition

    # step 2 make it into a distributed.RowMatrix
    coordMat = CoordinateMatrix(entries)

    coordMat = CoordinateMatrix(coordMat.entries.repartition(n_partitions))

    # this needs to be an IndexedRowMatrix()
    mat = coordMat.toRowMatrix()

    #goal: build a matrix of subreddit columns and tf-idfs rows
    sim_dist = mat.columnSimilarities(threshold=similarity_threshold)

    return (sim_dist, tfidf)


def build_tfidf_dataset(df, include_subs, term_colname, tf_family=tf_weight.Norm05):

    term = term_colname
    term_id = term + '_id'
    # aggregate counts by week. now subreddit-term is distinct
    df = df.filter(df.subreddit.isin(include_subs))
    df = df.groupBy(['subreddit',term]).agg(f.sum('tf').alias('tf'))

    max_subreddit_terms = df.groupby(['subreddit']).max('tf') # subreddits are unique
    max_subreddit_terms = max_subreddit_terms.withColumnRenamed('max(tf)','sr_max_tf')

    df = df.join(max_subreddit_terms, on='subreddit')

    df = df.withColumn("relative_tf", df.tf / df.sr_max_tf)

    # group by term. term is unique
    idf = df.groupby([term]).count()

    N_docs = df.select('subreddit').distinct().count()

    # add a little smoothing to the idf
    idf = idf.withColumn('idf',f.log(N_docs/(1+f.col('count')))+1)

    # collect the dictionary to make a pydict of terms to indexes
    terms = idf.select(term).distinct() # terms are distinct
    terms = terms.withColumn(term_id,f.row_number().over(Window.orderBy(term))) # term ids are distinct

    # make subreddit ids
    subreddits = df.select(['subreddit']).distinct()
    subreddits = subreddits.withColumn('subreddit_id',f.row_number().over(Window.orderBy("subreddit")))

    df = df.join(subreddits,on='subreddit')

    # map terms to indexes in the tfs and the idfs
    df = df.join(terms,on=term) # subreddit-term-id is unique

    idf = idf.join(terms,on=term)

    # join on subreddit/term to create tf/dfs indexed by term
    df = df.join(idf, on=[term_id, term])

    # agg terms by subreddit to make sparse tf/df vectors
    
    if tf_family == tf_weight.MaxTF:
        df = df.withColumn("tf_idf",  df.relative_tf * df.idf)
    else: # tf_fam = tf_weight.Norm05
        df = df.withColumn("tf_idf",  (0.5 + 0.5 * df.relative_tf) * df.idf)

    return df


