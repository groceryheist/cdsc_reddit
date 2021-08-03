import pandas as pd
import fire
from pathlib import Path
from similarities_helper import *
#from similarities_helper import similarities, lsi_column_similarities
from functools import partial

inpath = "/gscratch/comdata/users/nathante/competitive_exclusion_reddit/data/tfidf/comment_terms_compex.parquet/"
term_colname='term'
outfile='/gscratch/comdata/users/nathante/competitive_exclusion_reddit/data/similarity/comment_terms_compex_LSI'
n_components=[10,50,100]
included_subreddits="/gscratch/comdata/users/nathante/competitive_exclusion_reddit/data/included_subreddits.txt"
n_iter=5
random_state=1968
algorithm='arpack'
topN = None
from_date=None
to_date=None
min_df=None
max_df=None
def lsi_similarities(inpath, term_colname, outfile, min_df=None, max_df=None, included_subreddits=None, topN=None, from_date=None, to_date=None, tfidf_colname='tf_idf',n_components=100,n_iter=5,random_state=1968,algorithm='arpack',lsi_model=None):
    print(n_components,flush=True)

        
    if lsi_model is None:
        if type(n_components) == list:
            lsi_model = Path(outfile) / f'{max(n_components)}_{term_colname}_LSIMOD.pkl'
        else:
            lsi_model = Path(outfile) / f'{n_components}_{term_colname}_LSIMOD.pkl'

    simfunc = partial(lsi_column_similarities,n_components=n_components,n_iter=n_iter,random_state=random_state,algorithm=algorithm,lsi_model_save=lsi_model)

    return similarities(inpath=inpath, simfunc=simfunc, term_colname=term_colname, outfile=outfile, min_df=min_df, max_df=max_df, included_subreddits=included_subreddits, topN=topN, from_date=from_date, to_date=to_date, tfidf_colname=tfidf_colname)

# change so that these take in an input as an optional argument (for speed, but also for idf).
def term_lsi_similarities(inpath='/gscratch/comdata/output/reddit_similarity/tfidf/comment_terms_100k.parquet',outfile=None, min_df=None, max_df=None, included_subreddits=None, topN=None, from_date=None, to_date=None, algorithm='arpack', n_components=300,n_iter=5,random_state=1968):

    res =  lsi_similarities(inpath,
                            'term',
                            outfile,
                            min_df,
                            max_df,
                            included_subreddits,
                            topN,
                            from_date,
                            to_date,
                            n_components=n_components,
                            algorithm = algorithm
                            )
    return res

def author_lsi_similarities(inpath='/gscratch/comdata/output/reddit_similarity/tfidf/comment_authors_100k.parquet',outfile=None, min_df=2, max_df=None, included_subreddits=None, topN=None, from_date=None, to_date=None,algorithm='arpack',n_components=300,n_iter=5,random_state=1968):
    return lsi_similarities(inpath,
                            'author',
                            outfile,
                            min_df,
                            max_df,
                            included_subreddits,
                            topN,
                            from_date=from_date,
                            to_date=to_date,
                            n_components=n_components
                               )

def author_tf_similarities(inpath='/gscratch/comdata/output/reddit_similarity/tfidf/comment_authors_100k.parquet',outfile=None, min_df=2, max_df=None, included_subreddits=None, topN=None, from_date=None, to_date=None,n_components=300,n_iter=5,random_state=1968):
    return lsi_similarities(inpath,
                            'author',
                            outfile,
                            min_df,
                            max_df,
                            included_subreddits,
                            topN,
                            from_date=from_date,
                            to_date=to_date,
                            tfidf_colname='relative_tf',
                            n_components=n_components,
                            algorithm=algorithm
                            )


if __name__ == "__main__":
    fire.Fire({'term':term_lsi_similarities,
               'author':author_lsi_similarities,
               'author-tf':author_tf_similarities})

