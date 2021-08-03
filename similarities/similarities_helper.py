from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as f
from enum import Enum
from multiprocessing import cpu_count, Pool
from pyspark.mllib.linalg.distributed import CoordinateMatrix
from tempfile import TemporaryDirectory
import pyarrow
import pyarrow.dataset as ds
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix, issparse
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
import pathlib
from datetime import datetime
from pathlib import Path

class tf_weight(Enum):
    MaxTF = 1
    Norm05 = 2

infile = "/gscratch/comdata/output/reddit_similarity/tfidf_weekly/comment_terms.parquet"
cache_file = "/gscratch/comdata/users/nathante/cdsc_reddit/similarities/term_tfidf_entries_bak.parquet"

# subreddits missing after this step don't have any terms that have a high enough idf
# try rewriting without merges
def reindex_tfidf(infile, term_colname, min_df=None, max_df=None, included_subreddits=None, topN=500, week=None, from_date=None, to_date=None, rescale_idf=True, tf_family=tf_weight.MaxTF):
    print("loading tfidf", flush=True)
    tfidf_ds = ds.dataset(infile)

    if included_subreddits is None:
        included_subreddits = select_topN_subreddits(topN)
    else:
        included_subreddits = set(map(str.strip,map(str.lower,open(included_subreddits))))

    ds_filter = ds.field("subreddit").isin(included_subreddits)

    if min_df is not None:
        ds_filter &= ds.field("count") >= min_df

    if max_df is not None:
        ds_filter &= ds.field("count") <= max_df

    if week is not None:
        ds_filter &= ds.field("week") == week

    if from_date is not None:
        ds_filter &= ds.field("week") >= from_date

    if to_date is not None:
        ds_filter &= ds.field("week") <= to_date

    term = term_colname
    term_id = term + '_id'
    term_id_new = term + '_id_new'
    
    projection = {
        'subreddit_id':ds.field('subreddit_id'),
        term_id:ds.field(term_id),
        'relative_tf':ds.field("relative_tf").cast('float32')
        }

    if not rescale_idf:
        projection = {
            'subreddit_id':ds.field('subreddit_id'),
            term_id:ds.field(term_id),
            'relative_tf':ds.field('relative_tf').cast('float32'),
            'tf_idf':ds.field('tf_idf').cast('float32')}

    tfidf_ds = ds.dataset(infile)

    df = tfidf_ds.to_table(filter=ds_filter,columns=projection)

    df = df.to_pandas(split_blocks=True,self_destruct=True)
    print("assigning indexes",flush=True)
    df['subreddit_id_new'] = df.groupby("subreddit_id").ngroup()
    grouped = df.groupby(term_id)
    df[term_id_new] = grouped.ngroup()

    if rescale_idf:
        print("computing idf", flush=True)
        df['new_count'] = grouped[term_id].transform('count')
        N_docs = df.subreddit_id_new.max() + 1
        df['idf'] = np.log(N_docs/(1+df.new_count),dtype='float32') + 1
        if tf_family == tf_weight.MaxTF:
            df["tf_idf"] = df.relative_tf * df.idf
        else: # tf_fam = tf_weight.Norm05
            df["tf_idf"] = (0.5 + 0.5 * df.relative_tf) * df.idf

    print("assigning names")
    subreddit_names = tfidf_ds.to_table(filter=ds_filter,columns=['subreddit','subreddit_id'])
    batches = subreddit_names.to_batches()

    with Pool(cpu_count()) as pool:
        chunks = pool.imap_unordered(pull_names,batches) 
        subreddit_names = pd.concat(chunks,copy=False).drop_duplicates()

    subreddit_names = subreddit_names.set_index("subreddit_id")
    new_ids = df.loc[:,['subreddit_id','subreddit_id_new']].drop_duplicates()
    new_ids = new_ids.set_index('subreddit_id')
    subreddit_names = subreddit_names.join(new_ids,on='subreddit_id').reset_index()
    subreddit_names = subreddit_names.drop("subreddit_id",1)
    subreddit_names = subreddit_names.sort_values("subreddit_id_new")
    return(df, subreddit_names)

def pull_names(batch):
    return(batch.to_pandas().drop_duplicates())

def similarities(infile, simfunc, term_colname, outfile, min_df=None, max_df=None, included_subreddits=None, topN=500, from_date=None, to_date=None, tfidf_colname='tf_idf'):
    '''
    tfidf_colname: set to 'relative_tf' to use normalized term frequency instead of tf-idf, which can be useful for author-based similarities.
    '''

    def proc_sims(sims, outfile):
        if issparse(sims):
            sims = sims.todense()

        print(f"shape of sims:{sims.shape}")
        print(f"len(subreddit_names.subreddit.values):{len(subreddit_names.subreddit.values)}",flush=True)
        sims = pd.DataFrame(sims)
        sims = sims.rename({i:sr for i, sr in enumerate(subreddit_names.subreddit.values)}, axis=1)
        sims['_subreddit'] = subreddit_names.subreddit.values

        p = Path(outfile)

        output_feather =  Path(str(p).replace("".join(p.suffixes), ".feather"))
        output_csv =  Path(str(p).replace("".join(p.suffixes), ".csv"))
        output_parquet =  Path(str(p).replace("".join(p.suffixes), ".parquet"))
        outfile.parent.mkdir(exist_ok=True, parents=True)

        sims.to_feather(outfile)

    term = term_colname
    term_id = term + '_id'
    term_id_new = term + '_id_new'

    entries, subreddit_names = reindex_tfidf(infile, term_colname=term_colname, min_df=min_df, max_df=max_df, included_subreddits=included_subreddits, topN=topN,from_date=from_date,to_date=to_date)
    mat = csr_matrix((entries[tfidf_colname],(entries[term_id_new], entries.subreddit_id_new)))

    print("loading matrix")        

    #    mat = read_tfidf_matrix("term_tfidf_entries7ejhvnvl.parquet", term_colname)

    print(f'computing similarities on mat. mat.shape:{mat.shape}')
    print(f"size of mat is:{mat.data.nbytes}",flush=True)
    sims = simfunc(mat)
    del mat

    if hasattr(sims,'__next__'):
        for simmat, name in sims:
            proc_sims(simmat, Path(outfile)/(str(name) + ".feather"))
    else:
        proc_sims(simmat, outfile)

def write_weekly_similarities(path, sims, week, names):
    sims['week'] = week
    p = pathlib.Path(path)
    if not p.is_dir():
        p.mkdir(exist_ok=True,parents=True)
        
    # reformat as a pairwise list
    sims = sims.melt(id_vars=['_subreddit','week'],value_vars=names.subreddit.values)
    sims.to_parquet(p / week.isoformat())

def column_overlaps(mat):
    non_zeros = (mat != 0).astype('double')
    
    intersection = non_zeros.T @ non_zeros
    card1 = non_zeros.sum(axis=0)
    den = np.add.outer(card1,card1) - intersection

    return intersection / den
    
def test_lsi_sims():
    term = "term"
    term_id = term + '_id'
    term_id_new = term + '_id_new'

    t1 = time.perf_counter()
    entries, subreddit_names = reindex_tfidf("/gscratch/comdata/output/reddit_similarity/tfidf/comment_terms_100k_repartitioned.parquet",
                                             term_colname='term',
                                             min_df=2000,
                                             topN=10000
                                             )
    t2 = time.perf_counter()
    print(f"first load took:{t2 - t1}s")

    entries, subreddit_names = reindex_tfidf("/gscratch/comdata/output/reddit_similarity/tfidf/comment_terms_100k.parquet",
                                             term_colname='term',
                                             min_df=2000,
                                             topN=10000
                                             )
    t3=time.perf_counter()

    print(f"second load took:{t3 - t2}s")

    mat = csr_matrix((entries['tf_idf'],(entries[term_id_new], entries.subreddit_id_new)))
    sims = list(lsi_column_similarities(mat, [10,50]))
    sims_og = sims
    sims_test = list(lsi_column_similarities(mat,[10,50],algorithm='randomized',n_iter=10))

# n_components is the latent dimensionality. sklearn recommends 100. More might be better
# if n_components is a list we'll return a list of similarities with different latent dimensionalities
# if algorithm is 'randomized' instead of 'arpack' then n_iter gives the number of iterations.
# this function takes the svd and then the column similarities of it
def lsi_column_similarities(tfidfmat,n_components=300,n_iter=10,random_state=1968,algorithm='randomized'):
    # first compute the lsi of the matrix
    # then take the column similarities
    print("running LSI",flush=True)

    if type(n_components) is int:
        n_components = [n_components]

    n_components = sorted(n_components,reverse=True)
    
    svd_components = n_components[0]
    svd = TruncatedSVD(n_components=svd_components,random_state=random_state,algorithm=algorithm,n_iter=n_iter)
    mod = svd.fit(tfidfmat.T)
    lsimat = mod.transform(tfidfmat.T)
    for n_dims in n_components:
        sims = column_similarities(lsimat[:,np.arange(n_dims)])
        if len(n_components) > 1:
            yield (sims, n_dims)
        else:
            return sims
    

def column_similarities(mat):
    return 1 - pairwise_distances(mat,metric='cosine')


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

    df = df.repartition(400,'subreddit','week')
    dfwriter = df.write.partitionBy("week")
    return dfwriter

def _calc_tfidf(df, term_colname, tf_family):
    term = term_colname
    term_id = term + '_id'

    max_subreddit_terms = df.groupby(['subreddit']).max('tf') # subreddits are unique
    max_subreddit_terms = max_subreddit_terms.withColumnRenamed('max(tf)','sr_max_tf')

    df = df.join(max_subreddit_terms, on='subreddit')

    df = df.withColumn("relative_tf", (df.tf / df.sr_max_tf))

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
    df = df.repartition('subreddit')
    dfwriter = df.write
    return dfwriter

def select_topN_subreddits(topN, path="/gscratch/comdata/output/reddit_similarity/subreddits_by_num_comments_nonsfw.csv"):
    rankdf = pd.read_csv(path)
    included_subreddits = set(rankdf.loc[rankdf.comments_rank <= topN,'subreddit'].values)
    return included_subreddits


def repartition_tfidf(inpath="/gscratch/comdata/output/reddit_similarity/tfidf/comment_terms_100k.parquet",
                      outpath="/gscratch/comdata/output/reddit_similarity/tfidf/comment_terms_100k_repartitioned.parquet"):
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.parquet(inpath)
    df = df.repartition(400,'subreddit')
    df.write.parquet(outpath,mode='overwrite')

    
def repartition_tfidf_weekly(inpath="/gscratch/comdata/output/reddit_similarity/tfidf_weekly/comment_terms.parquet",
                      outpath="/gscratch/comdata/output/reddit_similarity/tfidf/comment_terms_repartitioned.parquet"):
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.parquet(inpath)
    df = df.repartition(400,'subreddit','week')
    dfwriter = df.write.partitionBy("week")
    dfwriter.parquet(outpath,mode='overwrite')
