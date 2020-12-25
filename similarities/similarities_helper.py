from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as f
from enum import Enum
from pyspark.mllib.linalg.distributed import CoordinateMatrix
from tempfile import TemporaryDirectory
import pyarrow
import pyarrow.dataset as ds
from scipy.sparse import csr_matrix, issparse
import pandas as pd
import numpy as np
import pathlib
from datetime import datetime
from pathlib import Path

class tf_weight(Enum):
    MaxTF = 1
    Norm05 = 2

infile = "/gscratch/comdata/output/reddit_similarity/tfidf_weekly/comment_authors.parquet"

def reindex_tfidf_time_interval(infile, term_colname, min_df=None, max_df=None, included_subreddits=None, topN=500, exclude_phrases=False, from_date=None, to_date=None):
    term = term_colname
    term_id = term + '_id'
    term_id_new = term + '_id_new'

    spark = SparkSession.builder.getOrCreate()
    conf = spark.sparkContext.getConf()
    print(exclude_phrases)
    tfidf_weekly = spark.read.parquet(infile)

    # create the time interval
    if from_date is not None:
        if type(from_date) is str:
            from_date = datetime.fromisoformat(from_date)

        tfidf_weekly = tfidf_weekly.filter(tfidf_weekly.week >= from_date)
        
    if to_date is not None:
        if type(to_date) is str:
            to_date = datetime.fromisoformat(to_date)
        tfidf_weekly = tfidf_weekly.filter(tfidf_weekly.week < to_date)

    tfidf = tfidf_weekly.groupBy(["subreddit","week", term_id, term]).agg(f.sum("tf").alias("tf"))
    tfidf = _calc_tfidf(tfidf, term_colname, tf_weight.Norm05)
    tempdir = prep_tfidf_entries(tfidf, term_colname, min_df, max_df, included_subreddits)
    tfidf = spark.read_parquet(tempdir.name)
    subreddit_names = tfidf.select(['subreddit','subreddit_id_new']).distinct().toPandas()
    subreddit_names = subreddit_names.sort_values("subreddit_id_new")
    subreddit_names['subreddit_id_new'] = subreddit_names['subreddit_id_new'] - 1
    return(tempdir, subreddit_names)

def reindex_tfidf(infile, term_colname, min_df=None, max_df=None, included_subreddits=None, topN=500, exclude_phrases=False):
    spark = SparkSession.builder.getOrCreate()
    conf = spark.sparkContext.getConf()
    print(exclude_phrases)

    tfidf = spark.read.parquet(infile)

    if included_subreddits is None:
        included_subreddits = select_topN_subreddits(topN)
    else:
        included_subreddits = set(open(included_subreddits))

    if exclude_phrases == True:
        tfidf = tfidf.filter(~f.col(term_colname).contains("_"))

    print("creating temporary parquet with matrix indicies")
    tempdir = prep_tfidf_entries(tfidf, term_colname, min_df, max_df, included_subreddits)

    tfidf = spark.read.parquet(tempdir.name)
    subreddit_names = tfidf.select(['subreddit','subreddit_id_new']).distinct().toPandas()
    subreddit_names = subreddit_names.sort_values("subreddit_id_new")
    subreddit_names['subreddit_id_new'] = subreddit_names['subreddit_id_new'] - 1
    spark.stop()
    return (tempdir, subreddit_names)

def similarities(infile, simfunc, term_colname, outfile, min_df=None, max_df=None, included_subreddits=None, topN=500, exclude_phrases=False, from_date=None, to_date=None):

    if from_date is not None or to_date is not None:
        tempdir, subreddit_names = reindex_tfidf_time_interval(infile, term_colname='author', min_df=min_df, max_df=max_df, included_subreddits=included_subreddits, topN=topN, exclude_phrases=False, from_date=from_date, to_date=to_date)
        
    else:
        tempdir, subreddit_names = reindex_tfidf(infile, term_colname='author', min_df=min_df, max_df=max_df, included_subreddits=included_subreddits, topN=topN, exclude_phrases=False)

    print("loading matrix")
    #    mat = read_tfidf_matrix("term_tfidf_entries7ejhvnvl.parquet", term_colname)
    mat = read_tfidf_matrix(tempdir.name, term_colname)
    print('computing similarities')
    sims = simfunc(mat)
    del mat

    if issparse(sims):
        sims = sims.todense()

    print(f"shape of sims:{sims.shape}")
    print(f"len(subreddit_names.subreddit.values):{len(subreddit_names.subreddit.values)}")
    sims = pd.DataFrame(sims)
    sims = sims.rename({i:sr for i, sr in enumerate(subreddit_names.subreddit.values)}, axis=1)
    sims['subreddit'] = subreddit_names.subreddit.values

    p = Path(outfile)

    output_feather =  Path(str(p).replace("".join(p.suffixes), ".feather"))
    output_csv =  Path(str(p).replace("".join(p.suffixes), ".csv"))
    output_parquet =  Path(str(p).replace("".join(p.suffixes), ".parquet"))

    sims.to_feather(outfile)
    tempdir.cleanup()

def read_tfidf_matrix_weekly(path, term_colname, week):
    term = term_colname
    term_id = term + '_id'
    term_id_new = term + '_id_new'

    dataset = ds.dataset(path,format='parquet')
    entries = dataset.to_table(columns=['tf_idf','subreddit_id_new',term_id_new],filter=ds.field('week')==week).to_pandas()
    return(csr_matrix((entries.tf_idf,(entries[term_id_new]-1, entries.subreddit_id_new-1))))

def write_weekly_similarities(path, sims, week, names):
    sims['week'] = week
    p = pathlib.Path(path)
    if not p.is_dir():
        p.mkdir()
        
    # reformat as a pairwise list
    sims = sims.melt(id_vars=['subreddit','week'],value_vars=names.subreddit.values)
    sims.to_parquet(p / week.isoformat())

def read_tfidf_matrix(path,term_colname):
    term = term_colname
    term_id = term + '_id'
    term_id_new = term + '_id_new'

    dataset = ds.dataset(path,format='parquet')
    entries = dataset.to_table(columns=['tf_idf','subreddit_id_new',term_id_new]).to_pandas()
    return(csr_matrix((entries.tf_idf,(entries[term_id_new]-1, entries.subreddit_id_new-1))))
    
def column_overlaps(mat):
    non_zeros = (mat != 0).astype('double')
    
    intersection = non_zeros.T @ non_zeros
    card1 = non_zeros.sum(axis=0)
    den = np.add.outer(card1,card1) - intersection

    return intersection / den
    
def column_similarities(mat):
    norm = np.matrix(np.power(mat.power(2).sum(axis=0),0.5,dtype=np.float32))
    mat = mat.multiply(1/norm)
    sims = mat.T @ mat
    return(sims)


def prep_tfidf_entries_weekly(tfidf, term_colname, min_df, max_df, included_subreddits):
    term = term_colname
    term_id = term + '_id'
    term_id_new = term + '_id_new'

    if min_df is None:
        min_df = 0.1 * len(included_subreddits)
        tfidf = tfidf.filter(f.col('count') >= min_df)
    if max_df is not None:
        tfidf = tfidf.filter(f.col('count') <= max_df)

    tfidf = tfidf.filter(f.col("subreddit").isin(included_subreddits))

    # we might not have the same terms or subreddits each week, so we need to make unique ids for each week.
    sub_ids = tfidf.select(['subreddit_id','week']).distinct()
    sub_ids = sub_ids.withColumn("subreddit_id_new",f.row_number().over(Window.partitionBy('week').orderBy("subreddit_id")))
    tfidf = tfidf.join(sub_ids,['subreddit_id','week'])

    # only use terms in at least min_df included subreddits in a given week
    new_count = tfidf.groupBy([term_id,'week']).agg(f.count(term_id).alias('new_count'))
    tfidf = tfidf.join(new_count,[term_id,'week'],how='inner')

    # reset the term ids
    term_ids = tfidf.select([term_id,'week']).distinct()
    term_ids = term_ids.withColumn(term_id_new,f.row_number().over(Window.partitionBy('week').orderBy(term_id)))
    tfidf = tfidf.join(term_ids,[term_id,'week'])

    tfidf = tfidf.withColumnRenamed("tf_idf","tf_idf_old")
    tfidf = tfidf.withColumn("tf_idf", (tfidf.relative_tf * tfidf.idf).cast('float'))

    tempdir =TemporaryDirectory(suffix='.parquet',prefix='term_tfidf_entries',dir='.')

    tfidf = tfidf.repartition('week')

    tfidf.write.parquet(tempdir.name,mode='overwrite',compression='snappy')
    return(tempdir)
    

def prep_tfidf_entries(tfidf, term_colname, min_df, max_df, included_subreddits):
    term = term_colname
    term_id = term + '_id'
    term_id_new = term + '_id_new'

    if min_df is None:
        min_df = 0.1 * len(included_subreddits)
        tfidf = tfidf.filter(f.col('count') >= min_df)
    if max_df is not None:
        tfidf = tfidf.filter(f.col('count') <= max_df)

    tfidf = tfidf.filter(f.col("subreddit").isin(included_subreddits))

    # reset the subreddit ids
    sub_ids = tfidf.select('subreddit_id').distinct()
    sub_ids = sub_ids.withColumn("subreddit_id_new", f.row_number().over(Window.orderBy("subreddit_id")))
    tfidf = tfidf.join(sub_ids,'subreddit_id')

    # only use terms in at least min_df included subreddits
    new_count = tfidf.groupBy(term_id).agg(f.count(term_id).alias('new_count'))
    tfidf = tfidf.join(new_count,term_id,how='inner')
    
    # reset the term ids
    term_ids = tfidf.select([term_id]).distinct()
    term_ids = term_ids.withColumn(term_id_new,f.row_number().over(Window.orderBy(term_id)))
    tfidf = tfidf.join(term_ids,term_id)

    tfidf = tfidf.withColumnRenamed("tf_idf","tf_idf_old")
    tfidf = tfidf.withColumn("tf_idf", (tfidf.relative_tf * tfidf.idf).cast('float'))
    
    tempdir =TemporaryDirectory(suffix='.parquet',prefix='term_tfidf_entries',dir='.')
    
    tfidf.write.parquet(tempdir.name,mode='overwrite',compression='snappy')
    return tempdir


# try computing cosine similarities using spark
def spark_cosine_similarities(tfidf, term_colname, min_df, included_subreddits, similarity_threshold):
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
    tfidf = tfidf.join(new_count,term_id,how='inner')
    
    # reset the term ids
    term_ids = tfidf.select([term_id]).distinct()
    term_ids = term_ids.withColumn(term_id_new,f.row_number().over(Window.orderBy(term_id)))
    tfidf = tfidf.join(term_ids,term_id)

    tfidf = tfidf.withColumnRenamed("tf_idf","tf_idf_old")
    tfidf = tfidf.withColumn("tf_idf", tfidf.relative_tf * tfidf.idf)

    # step 1 make an rdd of entires
    # sorted by (dense) spark subreddit id
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


def build_weekly_tfidf_dataset(df, include_subs, term_colname, tf_family=tf_weight.Norm05):
    term = term_colname
    term_id = term + '_id'

    # aggregate counts by week. now subreddit-term is distinct
    df = df.filter(df.subreddit.isin(include_subs))
    df = df.groupBy(['subreddit',term,'week']).agg(f.sum('tf').alias('tf'))

    max_subreddit_terms = df.groupby(['subreddit','week']).max('tf') # subreddits are unique
    max_subreddit_terms = max_subreddit_terms.withColumnRenamed('max(tf)','sr_max_tf')
    df = df.join(max_subreddit_terms, on=['subreddit','week'])
    df = df.withColumn("relative_tf", df.tf / df.sr_max_tf)

    # group by term. term is unique
    idf = df.groupby([term,'week']).count()

    N_docs = df.select(['subreddit','week']).distinct().groupby(['week']).agg(f.count("subreddit").alias("subreddits_in_week"))

    idf = idf.join(N_docs, on=['week'])

    # add a little smoothing to the idf
    idf = idf.withColumn('idf',f.log(idf.subreddits_in_week) / (1+f.col('count'))+1)

    # collect the dictionary to make a pydict of terms to indexes
    terms = idf.select([term,'week']).distinct() # terms are distinct

    terms = terms.withColumn(term_id,f.row_number().over(Window.partitionBy('week').orderBy(term))) # term ids are distinct

    # make subreddit ids
    subreddits = df.select(['subreddit','week']).distinct()
    subreddits = subreddits.withColumn('subreddit_id',f.row_number().over(Window.partitionBy("week").orderBy("subreddit")))

    df = df.join(subreddits,on=['subreddit','week'])

    # map terms to indexes in the tfs and the idfs
    df = df.join(terms,on=[term,'week']) # subreddit-term-id is unique

    idf = idf.join(terms,on=[term,'week'])

    # join on subreddit/term to create tf/dfs indexed by term
    df = df.join(idf, on=[term_id, term,'week'])

    # agg terms by subreddit to make sparse tf/df vectors
    
    if tf_family == tf_weight.MaxTF:
        df = df.withColumn("tf_idf",  df.relative_tf * df.idf)
    else: # tf_fam = tf_weight.Norm05
        df = df.withColumn("tf_idf",  (0.5 + 0.5 * df.relative_tf) * df.idf)

    return df

def _calc_tfidf(df, term_colname, tf_family):
    term = term_colname
    term_id = term + '_id'

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
    

def build_tfidf_dataset(df, include_subs, term_colname, tf_family=tf_weight.Norm05):
    term = term_colname
    term_id = term + '_id'
    # aggregate counts by week. now subreddit-term is distinct
    df = df.filter(df.subreddit.isin(include_subs))
    df = df.groupBy(['subreddit',term]).agg(f.sum('tf').alias('tf'))

    df = _calc_tfidf(df, term_colname, tf_family)

    return df

def select_topN_subreddits(topN, path="/gscratch/comdata/output/reddit_similarity/subreddits_by_num_comments.csv"):
    rankdf = pd.read_csv(path)
    included_subreddits = set(rankdf.loc[rankdf.comments_rank <= topN,'subreddit'].values)
    return included_subreddits
