from pyspark.sql import functions as f
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.mllib.linalg.distributed import RowMatrix, CoordinateMatrix
import numpy as np
import pyarrow
import pandas as pd
import fire
from itertools import islice
from pathlib import Path

min_df = 1000

spark = SparkSession.builder.getOrCreate()
conf = spark.sparkContext.getConf()

# outfile = '/gscratch/comdata/users/nathante/test_similarities_500.feather'; min_df = None; included_subreddits=None; similarity_threshold=0;
def spark_similarities(outfile, min_df = None, included_subreddits=None, similarity_threshold=0):
    '''
    Compute similarities between subreddits based on tfi-idf vectors of comment texts 
    
    included_subreddits : string
        Text file containing a list of subreddits to include (one per line) if included_subreddits is None then do the top 500 subreddits

    similarity_threshold : double (default = 0)
        set > 0 for large numbers of subreddits to get an approximate solution using the DIMSUM algorithm
https://stanford.edu/~rezab/papers/dimsum.pdf. If similarity_threshold=0 we get an exact solution using an O(N^2) algorithm.

    min_df : int (default = 0.1 * (number of included_subreddits)
         exclude terms that appear in fewer than this number of documents.

    outfile: string
         where to output csv and feather outputs
'''

    tfidf = spark.read.parquet('/gscratch/comdata/users/nathante/subreddit_tfidf.parquet')

    if included_subreddits is None:
        included_subreddits = list(islice(open("/gscratch/comdata/users/nathante/cdsc-reddit/top_25000_subs_by_comments.txt"),500))
        included_subreddits = [s.strip('\n') for s in included_subreddits]

    else:
        included_subreddits = set(open(included_subreddits))

    if min_df is None:
        min_df = 0.1 * len(included_subreddits)

    tfidf = tfidf.filter(f.col("subreddit").isin(included_subreddits))

    # reset the subreddit ids
    sub_ids = tfidf.select('subreddit_id').distinct()
    sub_ids = sub_ids.withColumn("subreddit_id_new",f.row_number().over(Window.orderBy("subreddit_id")))
    tfidf = tfidf.join(sub_ids,'subreddit_id')

    # only use terms in at least min_df included subreddits
    new_count = tfidf.groupBy('term_id').agg(f.count('term_id').alias('new_count'))
    term_ids = term_ids.join(new_count,'term_id')
    term_ids = term_ids.filter(new_count >= min_df)

    # reset the term ids
    term_ids = tfidf.select('term_id').distinct()
    term_ids = term_ids.withColumn("term_id_new",f.row_number().over(Window.orderBy("term_id")))
    tfidf = tfidf.join(term_ids,'term_id')

    # step 1 make an rdd of entires
    # sorted by (dense) spark subreddit id
    entries = tfidf.select(f.col("term_id_new")-1,f.col("subreddit_id_new")-1,"tf_idf").rdd

    # step 2 make it into a distributed.RowMatrix
    coordMat = CoordinateMatrix(entries)

    # this needs to be an IndexedRowMatrix()
    mat = coordMat.toRowMatrix()

    #goal: build a matrix of subreddit columns and tf-idfs rows
    sim_dist = mat.columnSimilarities(threshold=similarity_threshold)

    print(sim_dist.numRows(), sim_dist.numCols())

    #instead of toLocalMatrix() why not read as entries and put strait into numpy
    sim_entries = sim_dist.entries.collect()

    sim_entries = pd.DataFrame([{'i':me.i,'j':me.j,'value':me.value} for me in sim_entries])

    df = tfidf.select('subreddit','subreddit_id_new').distinct().toPandas()

    df = df.sort_values('subreddit_id_new').reset_index(drop=True)

    df = df.set_index('subreddit_id_new')

    similarities = sim_entries.join(df, on='i')
    similarities = sim_entries.rename(columns={'subreddit':"subreddit_i"})
    similarities = sim_entries.join(df, on='j')
    similarities = sim_entries.rename(columns={'subreddit':"subreddit_j"})

    p = Path(outfile)
    output_feather =  Path(str(p).replace("".join(p.suffixes), ".feather"))
    output_csv =  Path(str(p).replace("".join(p.suffixes), ".csv"))

    pyarrow.write_feather(similarities,output_feather)
    pyarrow.write_csv(similarities,output_csv)
    return similarities
    
if __name__ == '__main__':
    fire.Fire(spark_similarities)
