import fire
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from similarities_helper import build_tfidf_dataset, build_weekly_tfidf_dataset, select_topN_subreddits


def _tfidf_wrapper(func, inpath, outpath, topN, term_colname, exclude):
    spark = SparkSession.builder.getOrCreate()

    df = spark.read.parquet(inpath)

    df = df.filter(~ f.col(term_colname).isin(exclude))

    include_subs = select_topN_subreddits(topN)

    df = func(df, include_subs, term_colname)

    df.write.parquet(outpath,mode='overwrite',compression='snappy')

    spark.stop()

def tfidf(inpath, outpath, topN, term_colname, exclude):
    return _tfidf_wrapper(build_tfidf_dataset, inpath, outpath, topN, term_colname, exclude)

def tfidf_weekly(inpath, outpath, topN, term_colname, exclude):
    return _tfidf_wrapper(build_weekly_tfidf_dataset, inpath, outpath, topN, term_colname, exclude)

def tfidf_authors(outpath='/gscratch/comdata/output/reddit_similarity/tfidf/comment_authors.parquet',
                  topN=25000):

    return tfidf("/gscratch/comdata/output/reddit_ngrams/comment_authors.parquet",
                 outpath,
                 topN,
                 'author',
                 ['[deleted]','AutoModerator']
                 )

def tfidf_terms(outpath='/gscratch/comdata/output/reddit_similarity/tfidf/comment_terms.parquet',
                topN=25000):

    return tfidf("/gscratch/comdata/output/reddit_ngrams/comment_terms.parquet",
                 outpath,
                 topN,
                 'term',
                 []
                 )

def tfidf_authors_weekly(outpath='/gscratch/comdata/output/reddit_similarity/tfidf_weekly/comment_authors.parquet',
                  topN=25000):

    return tfidf_weekly("/gscratch/comdata/output/reddit_ngrams/comment_authors.parquet",
                 outpath,
                 topN,
                 'author',
                 ['[deleted]','AutoModerator']
                 )

def tfidf_terms_weekly(outpath='/gscratch/comdata/output/reddit_similarity/tfidf_weekly/comment_terms.parquet',
                topN=25000):


    return tfidf_weekly("/gscratch/comdata/output/reddit_ngrams/comment_terms.parquet",
                        outpath,
                        topN,
                        'term',
                        []
                        )


if __name__ == "__main__":
    fire.Fire({'authors':tfidf_authors,
               'terms':tfidf_terms,
               'authors_weekly':tfidf_authors_weekly,
               'terms_weekly':tfidf_terms_weekly})
