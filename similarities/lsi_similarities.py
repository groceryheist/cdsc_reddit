import pandas as pd
import fire
from pathlib import Path
from similarities_helper import similarities, lsi_column_similarities
from functools import partial

def lsi_similarities(infile, term_colname, outfile, min_df=None, max_df=None, included_subreddits=None, topN=500, from_date=None, to_date=None, tfidf_colname='tf_idf',n_components=100,n_iter=5,random_state=1968,algorithm='arpack'):
    print(n_components,flush=True)

    simfunc = partial(lsi_column_similarities,n_components=n_components,n_iter=n_iter,random_state=random_state,algorithm=algorithm)

    return similarities(infile=infile, simfunc=simfunc, term_colname=term_colname, outfile=outfile, min_df=min_df, max_df=max_df, included_subreddits=included_subreddits, topN=topN, from_date=from_date, to_date=to_date, tfidf_colname=tfidf_colname)

# change so that these take in an input as an optional argument (for speed, but also for idf).
def term_lsi_similarities(outfile, min_df=None, max_df=None, included_subreddits=None, topN=500, from_date=None, to_date=None, n_components=300,n_iter=5,random_state=1968,algorithm='arpack'):

    return lsi_similarities('/gscratch/comdata/output/reddit_similarity/tfidf/comment_terms_100k.parquet',
                            'term',
                            outfile,
                            min_df,
                            max_df,
                            included_subreddits,
                            topN,
                            from_date,
                            to_date,
                            n_components=n_components
                            )

def author_lsi_similarities(outfile, min_df=2, max_df=None, included_subreddits=None, topN=10000, from_date=None, to_date=None,n_components=300,n_iter=5,random_state=1968,algorithm='arpack'):
    return lsi_similarities('/gscratch/comdata/output/reddit_similarity/tfidf/comment_authors_100k.parquet',
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

def author_tf_similarities(outfile, min_df=2, max_df=None, included_subreddits=None, topN=10000, from_date=None, to_date=None,n_components=300,n_iter=5,random_state=1968,algorithm='arpack'):
    return lsi_similarities('/gscratch/comdata/output/reddit_similarity/tfidf/comment_authors_100k.parquet',
                            'author',
                            outfile,
                            min_df,
                            max_df,
                            included_subreddits,
                            topN,
                            from_date=from_date,
                            to_date=to_date,
                            tfidf_colname='relative_tf',
                            n_components=n_components
                            )


if __name__ == "__main__":
    fire.Fire({'term':term_lsi_similarities,
               'author':author_lsi_similarities,
               'author-tf':author_tf_similarities})

