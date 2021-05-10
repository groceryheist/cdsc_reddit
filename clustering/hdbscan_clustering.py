from clustering_base import clustering_result, clustering_job
from grid_sweep import grid_sweep
from dataclasses import dataclass
import hdbscan
from sklearn.neighbors import NearestNeighbors
import plotnine as pn
import numpy as np
from itertools import product, starmap, chain
import pandas as pd
from multiprocessing import cpu_count
import fire

def test_select_hdbscan_clustering():
    # select_hdbscan_clustering("/gscratch/comdata/output/reddit_similarity/subreddit_comment_authors-tf_30k_LSI",
    #                           "test_hdbscan_author30k",
    #                           min_cluster_sizes=[2],
    #                           min_samples=[1,2],
    #                           cluster_selection_epsilons=[0,0.05,0.1,0.15],
    #                           cluster_selection_methods=['eom','leaf'],
    #                           lsi_dimensions='all')
    inpath = "/gscratch/comdata/output/reddit_similarity/subreddit_comment_authors-tf_10k_LSI/"
    outpath = "test_hdbscan";
    min_cluster_sizes=[2,3,4];
    min_samples=[1,2,3];
    cluster_selection_epsilons=[0,0.1,0.3,0.5];
    cluster_selection_methods=['eom'];
    lsi_dimensions='all'
    gs = hdbscan_lsi_grid_sweep(inpath, "all", outpath, min_cluster_sizes, min_samples, cluster_selection_epsilons, cluster_selection_methods)
    gs.run(20)
    gs.save("test_hdbscan/lsi_sweep.csv")
    # job1 = hdbscan_lsi_job(infile=inpath, outpath=outpath, name="test", lsi_dims=500, min_cluster_size=2, min_samples=1,cluster_selection_epsilon=0,cluster_selection_method='eom')
    # job1.run()
    # print(job1.get_info())

    # df = pd.read_csv("test_hdbscan/selection_data.csv")
    # test_select_hdbscan_clustering()
    # check_clusters = pd.read_feather("test_hdbscan/500_2_2_0.1_eom.feather")
    # silscores = pd.read_feather("test_hdbscan/silhouette_samples500_2_2_0.1_eom.feather")
    # c = check_clusters.merge(silscores,on='subreddit')#    fire.Fire(select_hdbscan_clustering)
class hdbscan_grid_sweep(grid_sweep):
    def __init__(self,
                 inpath,
                 outpath,
                 *args,
                 **kwargs):

        super().__init__(hdbscan_job, inpath, outpath, self.namer, *args, **kwargs)

    def namer(self,
              min_cluster_size,
              min_samples,
              cluster_selection_epsilon,
              cluster_selection_method):
        return f"mcs-{min_cluster_size}_ms-{min_samples}_cse-{cluster_selection_epsilon}_csm-{cluster_selection_method}"

@dataclass
class hdbscan_clustering_result(clustering_result):
    min_cluster_size:int
    min_samples:int
    cluster_selection_epsilon:float
    cluster_selection_method:str

class hdbscan_job(clustering_job):
    def __init__(self, infile, outpath, name, min_cluster_size=2, min_samples=1, cluster_selection_epsilon=0, cluster_selection_method='eom'):
        super().__init__(infile,
                         outpath,
                         name,
                         call=hdbscan_job._hdbscan_clustering,
                         min_cluster_size=min_cluster_size,
                         min_samples=min_samples,
                         cluster_selection_epsilon=cluster_selection_epsilon,
                         cluster_selection_method=cluster_selection_method
                         )

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_method = cluster_selection_method
#        self.mat = 1 - self.mat

    def _hdbscan_clustering(mat, *args, **kwargs):
        print(f"running hdbscan clustering. args:{args}. kwargs:{kwargs}")
        print(mat)
        clusterer = hdbscan.HDBSCAN(metric='precomputed',
                                    core_dist_n_jobs=cpu_count(),
                                    *args,
                                    **kwargs,
                                    )
    
        clustering = clusterer.fit(mat.astype('double'))
    
        return(clustering)

    def get_info(self):
        result = super().get_info()
        self.result = hdbscan_clustering_result(**result.__dict__,
                                                min_cluster_size=self.min_cluster_size,
                                                min_samples=self.min_samples,
                                                cluster_selection_epsilon=self.cluster_selection_epsilon,
                                                cluster_selection_method=self.cluster_selection_method)
        return self.result

def run_hdbscan_grid_sweep(savefile, inpath, outpath,  min_cluster_sizes=[2], min_samples=[1], cluster_selection_epsilons=[0], cluster_selection_methods=['eom']):
    """Run hdbscan clustering once or more with different parameters.
    
    Usage:
    hdbscan_clustering.py --savefile=SAVEFILE --inpath=INPATH --outpath=OUTPATH --min_cluster_sizes=<csv> --min_samples=<csv> --cluster_selection_epsilons=<csv> --cluster_selection_methods=<csv "eom"|"leaf">

    Keword arguments:
    savefile: path to save the metadata and diagnostics 
    inpath: path to feather data containing a labeled matrix of subreddit similarities.
    outpath: path to output fit kmeans clusterings.
    min_cluster_sizes: one or more integers indicating the minumum cluster size
    min_samples: one ore more integers indicating the minimum number of samples used in the algorithm
    cluster_selection_epsilon: one or more similarity thresholds for transition from dbscan to hdbscan
    cluster_selection_method: "eom" or "leaf" eom gives larger clusters. 
    """    
    obj = hdbscan_grid_sweep(inpath,
                             outpath,
                             map(int,min_cluster_sizes),
                             map(int,min_samples),
                             map(float,cluster_selection_epsilons),
                             map(float,cluster_selection_methods))
    obj.run()
    obj.save(savefile)

def KNN_distances_plot(mat,outname,k=2):
    nbrs = NearestNeighbors(n_neighbors=k,algorithm='auto',metric='precomputed').fit(mat)
    distances, indices = nbrs.kneighbors(mat)
    d2 = distances[:,-1]
    df = pd.DataFrame({'dist':d2})
    df = df.sort_values("dist",ascending=False)
    df['idx'] = np.arange(0,d2.shape[0]) + 1
    p = pn.qplot(x='idx',y='dist',data=df,geom='line') + pn.scales.scale_y_continuous(minor_breaks = np.arange(0,50)/50,
                                                                                      breaks = np.arange(0,10)/10)
    p.save(outname,width=16,height=10)
    
def make_KNN_plots():
    similarities = "/gscratch/comdata/output/reddit_similarity/subreddit_comment_terms_10k.feather"
    subreddits, mat = read_similarity_mat(similarities)
    mat = sim_to_dist(mat)

    KNN_distances_plot(mat,k=2,outname='terms_knn_dist2.png')

    similarities = "/gscratch/comdata/output/reddit_similarity/subreddit_comment_authors_10k.feather"
    subreddits, mat = read_similarity_mat(similarities)
    mat = sim_to_dist(mat)
    KNN_distances_plot(mat,k=2,outname='authors_knn_dist2.png')

    similarities = "/gscratch/comdata/output/reddit_similarity/subreddit_comment_authors-tf_10k.feather"
    subreddits, mat = read_similarity_mat(similarities)
    mat = sim_to_dist(mat)
    KNN_distances_plot(mat,k=2,outname='authors-tf_knn_dist2.png')

if __name__ == "__main__":
    fire.Fire(run_hdbscan_grid_sweep)
    
#    test_select_hdbscan_clustering()
    #fire.Fire(select_hdbscan_clustering)  
