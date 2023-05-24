import fire
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from similarities_helper import tfidf_dataset, build_weekly_tfidf_dataset, select_topN_subreddits
from functools import partial

inpath = '/gscratch/comdata/users/nathante/competitive_exclusion_reddit/data/tfidf/comment_authors_compex.parquet'
# include_terms is a path to a parquet file that contains a column of term_colname + '_id' to include.
def _tfidf_wrapper(func, inpath, outpath, topN, term_colname, exclude, included_subreddits, included_terms=None, min_df=None, max_df=None):
    spark = SparkSession.builder.getOrCreate()

    df = spark.read.parquet(inpath)

    df = df.filter(~ f.col(term_colname).isin(exclude))

    if included_subreddits is not None:
        include_subs = set(map(str.strip,open(included_subreddits)))
    else:
        include_subs = select_topN_subreddits(topN)

    include_subs = spark.sparkContext.broadcast(include_subs)

    #    term_id = term_colname + "_id"

    if included_terms is not None:
        terms_df = spark.read.parquet(included_terms)
        terms_df = terms_df.select(term_colname).distinct()
        df = df.join(terms_df, on=term_colname, how='left_semi')

    dfwriter = func(df, include_subs.value, term_colname)

    dfwriter.parquet(outpath,mode='overwrite',compression='snappy')
    spark.stop()

def tfidf(inpath, outpath, topN, term_colname, exclude, included_subreddits, min_df, max_df):
    tfidf_func = partial(tfidf_dataset, max_df=max_df, min_df=min_df)
    return _tfidf_wrapper(tfidf_func, inpath, outpath, topN, term_colname, exclude, included_subreddits)

def tfidf_weekly(inpath, outpath, static_tfidf_path, topN, term_colname, exclude, included_subreddits):
    return _tfidf_wrapper(build_weekly_tfidf_dataset, inpath, outpath, topN, term_colname, exclude, included_subreddits, included_terms=static_tfidf_path)


def tfidf_authors(inpath="/gscratch/comdata/output/reddit_ngrams/comment_authors.parquet",
                  outpath='/gscratch/comdata/output/reddit_similarity/tfidf/comment_authors.parquet',
                  topN=None,
                  included_subreddits=None,
                  min_df=None,
                  max_df=None):

    return tfidf(inpath,
                 outpath,
                 topN,
                 'author',
                 ['[deleted]','AutoModerator'],
                 included_subreddits=included_subreddits,
                 min_df=min_df,
                 max_df=max_df
                 )

def tfidf_terms(inpath="/gscratch/comdata/output/reddit_ngrams/comment_terms.parquet",
                outpath='/gscratch/comdata/output/reddit_similarity/tfidf/comment_terms.parquet',
                topN=None,
                included_subreddits=None,
                min_df=None,
                max_df=None):

    return tfidf(inpath,
                 outpath,
                 topN,
                 'term',
                 [],
                 included_subreddits=included_subreddits,
                 min_df=min_df,
                 max_df=max_df
                 )

def tfidf_authors_weekly(inpath="/gscratch/comdata/output/reddit_ngrams/comment_authors.parquet",
                         static_tfidf_path="/gscratch/comdata/output/reddit_similarity/tfidf/comment_authors.parquet",
                         outpath='/gscratch/comdata/output/reddit_similarity/tfidf_weekly/comment_authors.parquet',
                         topN=None,
                         included_subreddits=None):

    return tfidf_weekly(inpath,
                        outpath,
                        static_tfidf_path,
                        topN,
                        'author',
                        ['[deleted]','AutoModerator'],
                        included_subreddits=included_subreddits
                        )

def tfidf_terms_weekly(inpath="/gscratch/comdata/output/reddit_ngrams/comment_terms.parquet",
                       static_tfidf_path="/gscratch/comdata/output/reddit_similarity/tfidf/comment_terms.parquet",
                       outpath='/gscratch/comdata/output/reddit_similarity/tfidf_weekly/comment_terms.parquet',
                       topN=None,
                       included_subreddits=None):


    return tfidf_weekly(inpath,
                        outpath,
                        static_tfidf_path,
                        topN,
                        'term',
                        [],
                        included_subreddits=included_subreddits
                        )


if __name__ == "__main__":
    fire.Fire({'authors':tfidf_authors,
               'terms':tfidf_terms,
               'authors_weekly':tfidf_authors_weekly,
               'terms_weekly':tfidf_terms_weekly})
