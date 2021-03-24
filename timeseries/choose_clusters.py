from pyarrow import dataset as ds
import numpy as np
import pandas as pd
import plotnine as pn
random = np.random.RandomState(1968)

def load_densities(term_density_file="/gscratch/comdata/output/reddit_density/comment_terms_10000.feather",
                   author_density_file="/gscratch/comdata/output/reddit_density/comment_authors_10000.feather"):

    term_density = pd.read_feather(term_density_file)
    author_density = pd.read_feather(author_density_file)

    term_density.rename({'overlap_density':'term_density','index':'subreddit'},axis='columns',inplace=True)
    author_density.rename({'overlap_density':'author_density','index':'subreddit'},axis='columns',inplace=True)

    density = term_density.merge(author_density,on='subreddit',how='inner')

    return density

def load_clusters(term_clusters_file="/gscratch/comdata/output/reddit_clustering/comment_terms_10000.feather",
                  author_clusters_file="/gscratch/comdata/output/reddit_clustering/comment_authors_10000.feather"):
    term_clusters = pd.read_feather(term_clusters_file)
    author_clusters = pd.read_feather(author_clusters_file)

    # rename, join and return
    term_clusters.rename({'cluster':'term_cluster'},axis='columns',inplace=True)
    author_clusters.rename({'cluster':'author_cluster'},axis='columns',inplace=True)

    clusters = term_clusters.merge(author_clusters,on='subreddit',how='inner')

    return clusters

if __name__ == '__main__':

    df = load_densities()
    cl = load_clusters()

    df['td_rank'] = df.term_density.rank()
    df['ad_rank'] = df.author_density.rank()

    df['td_percentile'] = df.td_rank / df.shape[0]
    df['ad_percentile'] = df.ad_rank / df.shape[0]

    df = df.merge(cl, on='subreddit',how='inner')

    term_cluster_density = df.groupby('term_cluster').agg({'td_rank':['mean','min','max'],
                                                         'ad_rank':['mean','min','max'],
                                                         'td_percentile':['mean','min','max'],
                                                           'ad_percentile':['mean','min','max'],
                                                           'subreddit':['count']})
                                                         

    author_cluster_density = df.groupby('author_cluster').agg({'td_rank':['mean','min','max'],
                                                         'ad_rank':['mean','min','max'],
                                                         'td_percentile':['mean','min','max'],
                                                           'ad_percentile':['mean','min','max'],
                                                           'subreddit':['count']})
                                                         
    # which clusters have the most term_density?
    term_cluster_density.iloc[term_cluster_density.td_rank['mean'].sort_values().index]

    # which clusters have the most author_density?
    term_cluster_density.iloc[term_cluster_density.ad_rank['mean'].sort_values(ascending=False).index].loc[term_cluster_density.subreddit['count'] >= 5][0:20]

    high_density_term_clusters = term_cluster_density.loc[(term_cluster_density.td_percentile['mean'] > 0.75) & (term_cluster_density.subreddit['count'] > 5)]

    # let's just use term density instead of author density for now. We can do a second batch with author density next.
    chosen_clusters = high_density_term_clusters.sample(3,random_state=random)

    cluster_info = df.loc[df.term_cluster.isin(chosen_clusters.index.values)]

    chosen_subreddits = cluster_info.subreddit.values

    dataset = ds.dataset("/gscratch/comdata/output/reddit_comments_by_subreddit.parquet",format='parquet')
    comments = dataset.to_table(filter=ds.field("subreddit").isin(chosen_subreddits),columns=['id','subreddit','author','CreatedAt'])

    comments = comments.to_pandas()

    comments['week'] = comments.CreatedAt.dt.date - pd.to_timedelta(comments['CreatedAt'].dt.dayofweek, unit='d')

    author_timeseries = comments.loc[:,['subreddit','author','week']].drop_duplicates().groupby(['subreddit','week']).count().reset_index()

    for clid in chosen_clusters.index.values:

        ts = pd.read_feather(f"data/ts_term_cluster_{clid}.feather")

        pn.options.figure_size = (11.7,8.27)
        p = pn.ggplot(ts)
        p = p + pn.geom_line(pn.aes('week','value',group='subreddit'))
        p = p + pn.facet_wrap('~ subreddit')
        p.save(f"plots/ts_term_cluster_{clid}.png")
        

        fig, ax = pyplot.subplots(figsize=(11.7,8.27))
        g = sns.FacetGrid(ts,row='subreddit')
        g.map_dataframe(sns.scatterplot,'week','value',data=ts,ax=ax)
