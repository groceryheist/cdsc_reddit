#!/usr/bin/env python3
import fire
import pandas as pd
from pathlib import Path
import shutil
selection_data="/gscratch/comdata/users/nathante/competitive_exclusion_reddit/data/clustering/comment_authors_compex_LSI/selection_data.csv"

outpath = 'test_best.feather'
min_clusters=50; max_isolates=7500; min_cluster_size=2

# pick the best clustering according to silhouette score subject to contraints
def pick_best_clustering(selection_data, output, min_clusters, max_isolates, min_cluster_size):
    df = pd.read_csv(selection_data,index_col=0)
    df = df.sort_values("silhouette_score",ascending=False)

    # not sure I fixed the bug underlying this fully or not.
    df['n_isolates_str'] = df.n_isolates.str.strip("[]")
    df['n_isolates_0'] = df['n_isolates_str'].apply(lambda l: len(l) == 0)
    df.loc[df.n_isolates_0,'n_isolates'] = 0
    df.loc[~df.n_isolates_0,'n_isolates'] = df.loc[~df.n_isolates_0].n_isolates_str.apply(lambda l: int(l))
    
    best_cluster = df[(df.n_isolates <= max_isolates)&(df.n_clusters >= min_clusters)&(df.min_cluster_size==min_cluster_size)]

    best_cluster = best_cluster.iloc[0]

    best_lsi_dimensions = best_cluster.lsi_dimensions
    print(best_cluster.to_dict())
    best_path = Path(best_cluster.outpath) / (str(best_cluster['name']) + ".feather")
    shutil.copy(best_path,output)
    print(f"lsi dimensions:{best_lsi_dimensions}")
    
if __name__ == "__main__":
    fire.Fire(pick_best_clustering)
