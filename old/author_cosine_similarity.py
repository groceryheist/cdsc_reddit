from pyspark.sql import functions as f
from pyspark.sql import SparkSession
from pyspark.sql import Window
import numpy as np
import pyarrow
import pandas as pd
import fire
from itertools import islice
from pathlib import Path
from similarities_helper import *

#tfidf = spark.read.parquet('/gscratch/comdata/output/reddit_similarity/tfidf_weekly/subreddit_terms.parquet')
def cosine_similarities_weekly(tfidf_path, outfile, term_colname, min_df = None, included_subreddits = None, topN = 500):
    spark = SparkSession.builder.getOrCreate()
    conf = spark.sparkContext.getConf()
    print(outfile)
    tfidf = spark.read.parquet(tfidf_path)
    
    if included_subreddits is None:
        included_subreddits = select_topN_subreddits(topN)

    else:
        included_subreddits = set(open(included_subreddits))

    print("creating temporary parquet with matrix indicies")
    tempdir = prep_tfidf_entries_weekly(tfidf, term_colname, min_df, included_subreddits)

    tfidf = spark.read.parquet(tempdir.name)

    # the ids can change each week.
    subreddit_names = tfidf.select(['subreddit','subreddit_id_new','week']).distinct().toPandas()
    subreddit_names = subreddit_names.sort_values("subreddit_id_new")
    subreddit_names['subreddit_id_new'] = subreddit_names['subreddit_id_new'] - 1
    spark.stop()

    weeks = list(subreddit_names.week.drop_duplicates())
    for week in weeks:
        print("loading matrix")
        mat = read_tfidf_matrix_weekly(tempdir.name, term_colname, week)
        print('computing similarities')
        sims = column_similarities(mat)
        del mat

        names = subreddit_names.loc[subreddit_names.week==week]

        sims = sims.rename({i:sr for i, sr in enumerate(names.subreddit.values)},axis=1)
        sims['subreddit'] = names.subreddit.values
        write_weekly_similarities(outfile, sims, week)



def cosine_similarities(outfile, min_df = None, included_subreddits=None, topN=500):
    '''
    Compute similarities between subreddits based on tfi-idf vectors of author comments
    
    included_subreddits : string
        Text file containing a list of subreddits to include (one per line) if included_subreddits is None then do the top 500 subreddits

    min_df : int (default = 0.1 * (number of included_subreddits)
         exclude terms that appear in fewer than this number of documents.

    outfile: string
         where to output csv and feather outputs
'''

    spark = SparkSession.builder.getOrCreate()
    conf = spark.sparkContext.getConf()
    print(outfile)

    tfidf = spark.read.parquet('/gscratch/comdata/output/reddit_similarity/tfidf/subreddit_comment_authors.parquet')

    if included_subreddits is None:
        included_subreddits = select_topN_subreddits(topN)

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
    
if __name__ == '__main__':
    fire.Fire(author_cosine_similarities)
