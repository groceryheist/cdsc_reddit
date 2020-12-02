from pyspark.sql import functions as f
from pyspark.sql import SparkSession
from pyspark.sql import Window
import numpy as np
import pyarrow
import pandas as pd
import fire
from itertools import islice
from pathlib import Path
from similarities_helper import cosine_similarities, prep_tfidf_entries, read_tfidf_matrix, column_similarities

spark = SparkSession.builder.getOrCreate()
conf = spark.sparkContext.getConf()

# outfile = '/gscratch/comdata/users/nathante/test_similarities_500.feather'; min_df = None; included_subreddits=None; similarity_threshold=0;
def author_cosine_similarities(outfile, min_df = None, included_subreddits=None, similarity_threshold=0, topN=500):
    '''
    Compute similarities between subreddits based on tfi-idf vectors of author comments
    
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

    spark = SparkSession.builder.getOrCreate()
    conf = spark.sparkContext.getConf()
    print(outfile)

    tfidf = spark.read.parquet('/gscratch/comdata/users/nathante/subreddit_tfidf_authors.parquet')

    if included_subreddits is None:
        rankdf = pd.read_csv("/gscratch/comdata/users/nathante/cdsc-reddit/subreddits_by_num_comments.csv")
        included_subreddits = set(rankdf.loc[rankdf.comments_rank <= topN,'subreddit'].values)

    else:
        included_subreddits = set(open(included_subreddits))

    print("creating temporary parquet with matrix indicies")
    tempdir = prep_tfidf_entries(tfidf, 'author', min_df, included_subreddits)
    tfidf = spark.read.parquet(tempdir.name)
    subreddit_names = tfidf.select(['subreddit','subreddit_id_new']).distinct().toPandas()
    subreddit_names = subreddit_names.sort_values("subreddit_id_new")
    subreddit_names['subreddit_id_new'] = subreddit_names['subreddit_id_new'] - 1
    spark.stop()

    print("loading matrix")
    mat = read_tfidf_matrix(tempdir.name,'author')
    print('computing similarities')
    sims = column_similarities(mat)
    del mat
    
    sims = pd.DataFrame(sims.todense())
    sims = sims.rename({i:sr for i, sr in enumerate(subreddit_names.subreddit.values)},axis=1)
    sims['subreddit'] = subreddit_names.subreddit.values

    p = Path(outfile)

    output_feather =  Path(str(p).replace("".join(p.suffixes), ".feather"))
    output_csv =  Path(str(p).replace("".join(p.suffixes), ".csv"))
    output_parquet =  Path(str(p).replace("".join(p.suffixes), ".parquet"))

    sims.to_feather(outfile)
    tempdir.cleanup()

    # print(outfile)

    # tfidf = spark.read.parquet('/gscratch/comdata/users/nathante/subreddit_tfidf_authors.parquet')

    # if included_subreddits is None:
    #     included_subreddits = list(islice(open("/gscratch/comdata/users/nathante/cdsc-reddit/top_25000_subs_by_comments.txt"),topN))
    #     included_subreddits = {s.strip('\n') for s in included_subreddits}

    # else:
    #     included_subreddits = set(open(included_subreddits))

    # sim_dist, tfidf = cosine_similarities(tfidf, 'author', min_df, included_subreddits, similarity_threshold)

    # p = Path(outfile)

    # output_feather =  Path(str(p).replace("".join(p.suffixes), ".feather"))
    # output_csv =  Path(str(p).replace("".join(p.suffixes), ".csv"))
    # output_parquet =  Path(str(p).replace("".join(p.suffixes), ".parquet"))
    # sim_dist = sim_dist.entries.toDF()

    # sim_dist = sim_dist.repartition(1)
    # sim_dist.write.parquet(str(output_parquet),mode='overwrite',compression='snappy')
    


    # #instead of toLocalMatrix() why not read as entries and put strait into numpy
    # sim_entries = pd.read_parquet(output_parquet)

    # df = tfidf.select('subreddit','subreddit_id_new').distinct().toPandas()

    # spark.stop()
    # df['subreddit_id_new'] = df['subreddit_id_new'] - 1
    # df = df.sort_values('subreddit_id_new').reset_index(drop=True)
    # df = df.set_index('subreddit_id_new')

    # similarities = sim_entries.join(df, on='i')
    # similarities = similarities.rename(columns={'subreddit':"subreddit_i"})
    # similarities = similarities.join(df, on='j')
    # similarities = similarities.rename(columns={'subreddit':"subreddit_j"})

    # similarities.to_feather(output_feather)
    # similarities.to_csv(output_csv)
    # return similarities
    
if __name__ == '__main__':
    fire.Fire(author_cosine_similarities)
