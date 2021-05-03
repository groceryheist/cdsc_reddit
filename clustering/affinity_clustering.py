from sklearn.metrics import silhouette_score
from sklearn.cluster import AffinityPropagation
from functools import partial
from dataclasses import dataclass
from clustering import _affinity_clustering, read_similarity_mat, sim_to_dist, process_clustering_result, clustering_result
from multiprocessing  import Pool, cpu_count, Array, Process
from pathlib import Path
from itertools import product, starmap
import numpy as np
import pandas as pd
import fire
import sys

# silhouette is the only one that doesn't need the feature matrix. So it's probably the only one that's worth trying. 
@dataclass
class affinity_clustering_result(clustering_result):
    damping:float
    convergence_iter:int
    preference_quantile:float

def affinity_clustering(similarities, output, *args, **kwargs):
    subreddits, mat = read_similarity_mat(similarities)
    clustering = _affinity_clustering(mat, *args, **kwargs)
    cluster_data = process_clustering_result(clustering, subreddits)
    cluster_data['algorithm'] = 'affinity'
    return(cluster_data)

def _affinity_clustering(mat, subreddits, output, damping=0.9, max_iter=100000, convergence_iter=30, preference_quantile=0.5, random_state=1968, verbose=True):
    '''
    similarities: matrix of similarity scores
    preference_quantile: parameter controlling how many clusters to make. higher values = more clusters. 0.85 is a good value with 3000 subreddits.
    damping: parameter controlling how iterations are merged. Higher values make convergence faster and more dependable. 0.85 is a good value for the 10000 subreddits by author. 
    '''
    print(f"damping:{damping}; convergenceIter:{convergence_iter}; preferenceQuantile:{preference_quantile}")

    preference = np.quantile(mat,preference_quantile)

    print(f"preference is {preference}")
    print("data loaded")
    sys.stdout.flush()
    clustering = AffinityPropagation(damping=damping,
                                     max_iter=max_iter,
                                     convergence_iter=convergence_iter,
                                     copy=False,
                                     preference=preference,
                                     affinity='precomputed',
                                     verbose=verbose,
                                     random_state=random_state).fit(mat)

    cluster_data = process_clustering_result(clustering, subreddits)
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    cluster_data.to_feather(output)
    print(f"saved {output}")
    return clustering


def do_clustering(damping, convergence_iter, preference_quantile, name, mat, subreddits,  max_iter,  outdir:Path, random_state, verbose, alt_mat, overwrite=False):
    if name is None:
        name = f"damping-{damping}_convergenceIter-{convergence_iter}_preferenceQuantile-{preference_quantile}"
    print(name)
    sys.stdout.flush()
    outpath = outdir / (str(name) + ".feather")
    outpath.parent.mkdir(parents=True,exist_ok=True)
    print(outpath)
    clustering = _affinity_clustering(mat, outpath, damping, max_iter, convergence_iter, preference_quantile, random_state, verbose)
    cluster_data = process_clustering_result(clustering, subreddits)
    mat = sim_to_dist(clustering.affinity_matrix_)

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
    
    res = affinity_clustering_result(outpath=outpath,
                                     damping=damping,
                                     max_iter=max_iter,
                                     convergence_iter=convergence_iter,
                                     preference_quantile=preference_quantile,
                                     silhouette_score=score,
                                     alt_silhouette_score=score,
                                     name=str(name))

    return res

# alt similiarities is for checking the silhouette coefficient of an alternative measure of similarity (e.g., topic similarities for user clustering).

def select_affinity_clustering(similarities, outdir, outinfo, damping=[0.9], max_iter=100000, convergence_iter=[30], preference_quantile=[0.5], random_state=1968, verbose=True, alt_similarities=None, J=None):

    damping = list(map(float,damping))
    convergence_iter = convergence_iter = list(map(int,convergence_iter))
    preference_quantile = list(map(float,preference_quantile))

    if type(outdir) is str:
        outdir = Path(outdir)

    outdir.mkdir(parents=True,exist_ok=True)

    subreddits, mat = read_similarity_mat(similarities,use_threads=True)

    if alt_similarities is not None:
        alt_mat = read_similarity_mat(alt_similarities,use_threads=True)
    else:
        alt_mat = None

    if J is None:
        J = cpu_count()
    pool = Pool(J)

    # get list of tuples: the combinations of hyperparameters
    hyper_grid = product(damping, convergence_iter, preference_quantile)
    hyper_grid = (t + (str(i),) for i, t in enumerate(hyper_grid))

    _do_clustering = partial(do_clustering,  mat=mat, subreddits=subreddits, outdir=outdir, max_iter=max_iter, random_state=random_state, verbose=verbose, alt_mat=alt_mat)

    #    similarities = Array('d', mat)
    # call pool.starmap
    print("running clustering selection")
    clustering_data = pool.starmap(_do_clustering, hyper_grid)
    clustering_data = pd.DataFrame(list(clustering_data))
    clustering_data.to_csv(outinfo)
    return clustering_data

if __name__ == "__main__":
    x = fire.Fire(select_affinity_clustering)
