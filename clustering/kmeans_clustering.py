from sklearn.cluster import KMeans
import fire
from pathlib import Path
from dataclasses import dataclass
from clustering_base import clustering_result, clustering_job
from grid_sweep import grid_sweep

@dataclass
class kmeans_clustering_result(clustering_result):
    n_clusters:int
    n_init:int
    max_iter:int

class kmeans_job(clustering_job):
    def __init__(self, infile, outpath, name, n_clusters, n_init=10, max_iter=100000, random_state=1968, verbose=True):
        super().__init__(infile,
                         outpath,
                         name,
                         call=kmeans_job._kmeans_clustering,
                         n_clusters=n_clusters,
                         n_init=n_init,
                         max_iter=max_iter,
                         random_state=random_state,
                         verbose=verbose)

        self.n_clusters=n_clusters
        self.n_init=n_init
        self.max_iter=max_iter

    def _kmeans_clustering(mat, *args, **kwargs):

        clustering = KMeans(*args,
                            **kwargs,
                            ).fit(mat)

        return clustering


    def get_info(self):
        result = super().get_info()
        self.result = kmeans_clustering_result(**result.__dict__,
                                               n_init=self.n_init,
                                               max_iter=self.max_iter)
        return self.result


class kmeans_grid_sweep(grid_sweep):
       
    def __init__(self,
                 inpath,
                 outpath,
                 *args,
                 **kwargs):
        super().__init__(kmeans_job, inpath, outpath, self.namer, *args, **kwargs)

    def namer(self,
             n_clusters,
             n_init,
             max_iter):
        return f"nclusters-{n_clusters}_nit-{n_init}_maxit-{max_iter}"

def test_select_kmeans_clustering():
    inpath = "/gscratch/comdata/output/reddit_similarity/subreddit_comment_authors-tf_10k_LSI/"
    outpath = "test_kmeans";
    n_clusters=[200,300,400];
    n_init=[1,2,3];
    max_iter=[100000]

    gs = kmeans_lsi_grid_sweep(inpath, 'all', outpath, n_clusters, n_init, max_iter)
    gs.run(1)

    cluster_selection_epsilons=[0,0.1,0.3,0.5];
    cluster_selection_methods=['eom'];
    lsi_dimensions='all'
    gs = hdbscan_lsi_grid_sweep(inpath, "all", outpath, min_cluster_sizes, min_samples, cluster_selection_epsilons, cluster_selection_methods)
    gs.run(20)
    gs.save("test_hdbscan/lsi_sweep.csv")

def run_kmeans_grid_sweep(savefile, inpath, outpath,  n_clusters=[500], n_inits=[1], max_iters=[3000]):
    """Run kmeans clustering once or more with different parameters.
    
    Usage:
    kmeans_clustering.py --savefile=SAVEFILE --inpath=INPATH --outpath=OUTPATH --n_clusters=<csv number of clusters> --n_inits=<csv> --max_iters=<csv>

    Keword arguments:
    savefile: path to save the metadata and diagnostics 
    inpath: path to feather data containing a labeled matrix of subreddit similarities.
    outpath: path to output fit kmeans clusterings.
    n_clusters: one or more numbers of kmeans clusters to select.
    n_inits: one or more numbers of different initializations to use for each clustering.
    max_iters: one or more numbers of different maximum interations. 
    """    

    obj = kmeans_grid_sweep(inpath,
                            outpath,
                            map(int,n_clusters),
                            map(int,n_inits),
                            map(int,max_iters))


    obj.run(1)
    obj.save(savefile)

if __name__ == "__main__":
    fire.Fire(run_kmeans_grid_sweep)
