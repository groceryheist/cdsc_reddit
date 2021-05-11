from sklearn.cluster import KMeans
import fire
from pathlib import Path
from multiprocessing import cpu_count
from dataclasses import dataclass
from clustering_base import sim_to_dist, process_clustering_result, clustering_result, read_similarity_mat
from clustering_base import lsi_result_mixin, lsi_mixin, clustering_job, grid_sweep, lsi_grid_sweep


@dataclass
class kmeans_clustering_result(clustering_result):
    n_clusters:int
    n_init:int
    max_iter:int

@dataclass
class kmeans_clustering_result_lsi(kmeans_clustering_result, lsi_result_mixin):
    pass

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
                                               n_init=n_init,
                                               max_iter=max_iter)
        return self.result


class kmeans_lsi_job(kmeans_job, lsi_mixin):
    def __init__(self, infile, outpath, name, lsi_dims, *args, **kwargs):
        super().__init__(infile,
                         outpath,
                         name,
                         *args,
                         **kwargs)
        super().set_lsi_dims(lsi_dims)

    def get_info(self):
        result = super().get_info()
        self.result = kmeans_clustering_result_lsi(**result.__dict__,
                                                   lsi_dimensions=self.lsi_dims)
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

class _kmeans_lsi_grid_sweep(grid_sweep):
    def __init__(self,
                 inpath,
                 outpath,
                 lsi_dim,
                 *args,
                 **kwargs):
        self.lsi_dim = lsi_dim
        self.jobtype = kmeans_lsi_job
        super().__init__(self.jobtype, inpath, outpath, self.namer, self.lsi_dim, *args, **kwargs)

    def namer(self, *args, **kwargs):
        s = kmeans_grid_sweep.namer(self, *args[1:], **kwargs)
        s += f"_lsi-{self.lsi_dim}"
        return s

class kmeans_lsi_grid_sweep(lsi_grid_sweep):
    def __init__(self,
                 inpath,
                 lsi_dims,
                 outpath,
                 n_clusters,
                 n_inits,
                 max_iters
                 ):

        super().__init__(kmeans_lsi_job,
                         _kmeans_lsi_grid_sweep,
                         inpath,
                         lsi_dims,
                         outpath,
                         n_clusters,
                         n_inits,
                         max_iters)

def test_select_kmeans_clustering():
    # select_hdbscan_clustering("/gscratch/comdata/output/reddit_similarity/subreddit_comment_authors-tf_30k_LSI",
    #                           "test_hdbscan_author30k",
    #                           min_cluster_sizes=[2],
    #                           min_samples=[1,2],
    #                           cluster_selection_epsilons=[0,0.05,0.1,0.15],
    #                           cluster_selection_methods=['eom','leaf'],
    #                           lsi_dimensions='all')
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


if __name__ == "__main__":

    fire.Fire{'grid_sweep':kmeans_grid_sweep,
              'grid_sweep_lsi':kmeans_lsi_grid_sweep
              'cluster':kmeans_job,
              'cluster_lsi':kmeans_lsi_job}
