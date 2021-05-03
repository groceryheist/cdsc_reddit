from sklearn.metrics import silhouette_score
from sklearn.cluster import AffinityPropagation
from functools import partial
from clustering import _kmeans_clustering, read_similarity_mat, sim_to_dist, process_clustering_result, clustering_result
from dataclasses import dataclass
from multiprocessing  import Pool, cpu_count, Array, Process
from pathlib import Path
from itertools import product, starmap
import numpy as np
import pandas as pd
import fire
import sys

@dataclass
class kmeans_clustering_result(clustering_result):
    n_clusters:int
    n_init:int


# silhouette is the only one that doesn't need the feature matrix. So it's probably the only one that's worth trying. 

def do_clustering(n_clusters, n_init, name, mat, subreddits,  max_iter, outdir:Path, random_state, verbose, alt_mat, overwrite=False):
    if name is None:
        name = f"damping-{damping}_convergenceIter-{convergence_iter}_preferenceQuantile-{preference_quantile}"
    print(name)
    sys.stdout.flush()
    outpath = outdir / (str(name) + ".feather")
    print(outpath)
    mat = sim_to_dist(mat)
    clustering = _kmeans_clustering(mat, outpath, n_clusters, n_init, max_iter, random_state, verbose)

    outpath.parent.mkdir(parents=True,exist_ok=True)
    cluster_data.to_feather(outpath)
    cluster_data = process_clustering_result(clustering, subreddits)

    try: 
        score = silhouette_score(mat, clustering.labels_, metric='precomputed')
    except ValueError:
        score = None

    if alt_mat is not None:
        alt_distances = sim_to_dist(alt_mat)
        try:
            alt_score = silhouette_score(alt_mat, clustering.labels_, metric='precomputed')
        except ValueError:
            alt_score = None
    
    res = kmeans_clustering_result(outpath=outpath,
                                   max_iter=max_iter,
                                   n_clusters=n_clusters,
                                   n_init = n_init,
                                   silhouette_score=score,
                                   alt_silhouette_score=score,
                                   name=str(name))

    return res


# alt similiarities is for checking the silhouette coefficient of an alternative measure of similarity (e.g., topic similarities for user clustering).
def select_kmeans_clustering(similarities, outdir, outinfo, n_clusters=[1000], max_iter=100000, n_init=10, random_state=1968, verbose=True, alt_similarities=None):

    n_clusters = list(map(int,n_clusters))
    n_init  = list(map(int,n_init))

    if type(outdir) is str:
        outdir = Path(outdir)

    outdir.mkdir(parents=True,exist_ok=True)

    subreddits, mat = read_similarity_mat(similarities,use_threads=True)

    if alt_similarities is not None:
        alt_mat = read_similarity_mat(alt_similarities,use_threads=True)
    else:
        alt_mat = None

    # get list of tuples: the combinations of hyperparameters
    hyper_grid = product(n_clusters, n_init)
    hyper_grid = (t + (str(i),) for i, t in enumerate(hyper_grid))

    _do_clustering = partial(do_clustering,  mat=mat, subreddits=subreddits, outdir=outdir, max_iter=max_iter, random_state=random_state, verbose=verbose, alt_mat=alt_mat)

    # call starmap
    print("running clustering selection")
    clustering_data = starmap(_do_clustering, hyper_grid)
    clustering_data = pd.DataFrame(list(clustering_data))
    clustering_data.to_csv(outinfo)
    
    return clustering_data

if __name__ == "__main__":
    x = fire.Fire(select_kmeans_clustering)
