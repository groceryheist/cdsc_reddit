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
from similarities_helper import prep_tfidf_entries, read_tfidf_matrix, column_similarities, select_topN
import scipy

# outfile='test_similarities_500.feather';
# min_df = None;
# included_subreddits=None; topN=100; exclude_phrases=True;
def term_cosine_similarities(outfile, min_df = None, included_subreddits=None, topN=500, exclude_phrases=False):
    spark = SparkSession.builder.getOrCreate()
    conf = spark.sparkContext.getConf()
    print(outfile)
    print(exclude_phrases)

    tfidf = spark.read.parquet('/gscratch/comdata/output/reddit_similarity/tfidf/subreddit_terms.parquet')

    if included_subreddits is None:
        included_subreddits = select_topN_subreddits(topN)
    else:
        included_subreddits = set(open(included_subreddits))

    if exclude_phrases == True:
        tfidf = tfidf.filter(~f.col(term).contains("_"))

    print("creating temporary parquet with matrix indicies")
    tempdir = prep_tfidf_entries(tfidf, 'term', min_df, included_subreddits)
    tfidf = spark.read.parquet(tempdir.name)
    subreddit_names = tfidf.select(['subreddit','subreddit_id_new']).distinct().toPandas()
    subreddit_names = subreddit_names.sort_values("subreddit_id_new")
    subreddit_names['subreddit_id_new'] = subreddit_names['subreddit_id_new'] - 1
    spark.stop()

    print("loading matrix")
    mat = read_tfidf_matrix(tempdir.name,'term')
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
    fire.Fire(term_cosine_similarities)
