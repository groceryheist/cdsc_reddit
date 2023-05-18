from clustering_base import clustering_result, clustering_job, twoway_clustering_job
from hdbscan_clustering import hdbscan_clustering_result
import umap
from grid_sweep import twoway_grid_sweep
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
    inpath = "/gscratch/comdata/output/reddit_similarity/subreddit_comment_authors-tf_10k_LSI"
    outpath = "test_umap_hdbscan_lsi"
    min_cluster_sizes=[2,3,4]
    min_samples=[1,2,3]
    cluster_selection_epsilons=[0,0.1,0.3,0.5]
    cluster_selection_methods=[1]
    lsi_dimensions='all'
    n_neighbors = [5,10,15,25,35,70,100]
    learning_rate = [0.1,0.5,1,2]
    min_dist = [0.5,1,1.5,2]
    local_connectivity = [1,2,3,4,5]

    hdbscan_params = {"min_cluster_sizes":min_cluster_sizes, "min_samples":min_samples, "cluster_selection_epsilons":cluster_selection_epsilons, "cluster_selection_methods":cluster_selection_methods}
    umap_params = {"n_neighbors":n_neighbors, "learning_rate":learning_rate, "min_dist":min_dist, "local_connectivity":local_connectivity}
    gs = umap_hdbscan_grid_sweep(inpath, "all", outpath, hdbscan_params,umap_params)

    # gs.run(20)
    # gs.save("test_hdbscan/lsi_sweep.csv")


    # job1 = hdbscan_lsi_job(infile=inpath, outpath=outpath, name="test", lsi_dims=500, min_cluster_size=2, min_samples=1,cluster_selection_epsilon=0,cluster_selection_method='eom')
    # job1.run()
    # print(job1.get_info())

    # df = pd.read_csv("test_hdbscan/selection_data.csv")
    # test_select_hdbscan_clustering()
    # check_clusters = pd.read_feather("test_hdbscan/500_2_2_0.1_eom.feather")
    # silscores = pd.read_feather("test_hdbscan/silhouette_samples500_2_2_0.1_eom.feather")
    # c = check_clusters.merge(silscores,on='subreddit')#    fire.Fire(select_hdbscan_clustering)
class umap_hdbscan_grid_sweep(twoway_grid_sweep):
    def __init__(self,
                 inpath,
                 outpath,
                 umap_params,
                 hdbscan_params):

        super().__init__(umap_hdbscan_job, inpath, outpath, self.namer, umap_params, hdbscan_params)

    def namer(self,
              min_cluster_size,
              min_samples,
              cluster_selection_epsilon,
              cluster_selection_method,
              n_components,
              n_neighbors,
              learning_rate,
              min_dist,
              local_connectivity,
              densmap
              ):
        return f"mcs-{min_cluster_size}_ms-{min_samples}_cse-{cluster_selection_epsilon}_csm-{cluster_selection_method}_nc-{n_components}_nn-{n_neighbors}_lr-{learning_rate}_md-{min_dist}_lc-{local_connectivity}_dm-{densmap}"

@dataclass
class umap_hdbscan_clustering_result(hdbscan_clustering_result):
    n_components:int
    n_neighbors:int
    learning_rate:float
    min_dist:float
    local_connectivity:int
    densmap:bool

class umap_hdbscan_job(twoway_clustering_job):
    def __init__(self, infile, outpath, name,
                 umap_args = {"n_components":2,"n_neighbors":15, "learning_rate":1, "min_dist":1, "local_connectivity":1,'densmap':False},
                 hdbscan_args = {"min_cluster_size":2, "min_samples":1, "cluster_selection_epsilon":0, "cluster_selection_method":'eom'},
                 *args,
                 **kwargs):
        super().__init__(infile,
                         outpath,
                         name,
                         call1=umap_hdbscan_job._umap_embedding,
                         call2=umap_hdbscan_job._hdbscan_clustering,
                         args1=umap_args,
                         args2=hdbscan_args,
                         *args,
                         **kwargs
                         )

        self.n_components = umap_args['n_components']
        self.n_neighbors = umap_args['n_neighbors']
        self.learning_rate = umap_args['learning_rate']
        self.min_dist = umap_args['min_dist']
        self.local_connectivity = umap_args['local_connectivity']
        self.densmap = umap_args['densmap']
        self.min_cluster_size = hdbscan_args['min_cluster_size']
        self.min_samples = hdbscan_args['min_samples']
        self.cluster_selection_epsilon = hdbscan_args['cluster_selection_epsilon']
        self.cluster_selection_method = hdbscan_args['cluster_selection_method']

    def after_run(self):
        coords = self.step1.embedding_
        self.cluster_data['x'] = coords[:,0]
        self.cluster_data['y'] = coords[:,1]
        super().after_run()


    def _umap_embedding(mat, **umap_args):
        print(f"running umap embedding. umap_args:{umap_args}")
        umapmodel = umap.UMAP(metric='precomputed', **umap_args)
        umapmodel = umapmodel.fit(mat)
        return umapmodel

    def _hdbscan_clustering(mat, umapmodel, **hdbscan_args):
        print(f"running hdbascan clustering. hdbscan_args:{hdbscan_args}")
        
        umap_coords = umapmodel.transform(mat)

        clusterer = hdbscan.HDBSCAN(metric='euclidean',
                                    core_dist_n_jobs=cpu_count(),
                                    **hdbscan_args
                                    )
    
        clustering = clusterer.fit(umap_coords)
    
        return(clustering)

    def get_info(self):
        result = super().get_info()
        self.result = umap_hdbscan_clustering_result(**result.__dict__,
                                                     min_cluster_size=self.min_cluster_size,
                                                     min_samples=self.min_samples,
                                                     cluster_selection_epsilon=self.cluster_selection_epsilon,
                                                     cluster_selection_method=self.cluster_selection_method,
                                                     n_components = self.n_components,
                                                     n_neighbors = self.n_neighbors,
                                                     learning_rate = self.learning_rate,
                                                     min_dist = self.min_dist,
                                                     local_connectivity=self.local_connectivity,
                                                     densmap=self.densmap
                                                     )
        return self.result

def run_umap_hdbscan_grid_sweep(savefile, inpath, outpath, n_neighbors = [15], n_components=[2], learning_rate=[1], min_dist=[1], local_connectivity=[1],
                                densmap=[False],
                                min_cluster_sizes=[2], min_samples=[1], cluster_selection_epsilons=[0], cluster_selection_methods=['eom']):
    """Run umap + hdbscan clustering once or more with different parameters.
    
    Usage:
    umap_hdbscan_clustering.py --savefile=SAVEFILE --inpath=INPATH --outpath=OUTPATH --n_neighbors=<csv> --learning_rate=<csv> --min_dist=<csv> --local_connectivity=<csv> --min_cluster_sizes=<csv> --min_samples=<csv> --cluster_selection_epsilons=<csv> --cluster_selection_methods=<csv "eom"|"leaf">

    Keword arguments:
    savefile: path to save the metadata and diagnostics 
    inpath: path to feather data containing a labeled matrix of subreddit similarities.
    outpath: path to output fit kmeans clusterings.
    n_neighbors: umap parameter takes integers greater than 1
    learning_rate: umap parameter takes positive real values
    min_dist: umap parameter takes positive real values
    local_connectivity: umap parameter takes positive integers
    min_cluster_sizes: one or more integers indicating the minumum cluster size
    min_samples: one ore more integers indicating the minimum number of samples used in the algorithm
    cluster_selection_epsilon: one or more similarity thresholds for transition from dbscan to hdbscan
    cluster_selection_method: "eom" or "leaf" eom gives larger clusters. 
    """    
    
    umap_args = {'n_neighbors':list(map(int, n_neighbors)),
                 'learning_rate':list(map(float,learning_rate)),
                 'min_dist':list(map(float,min_dist)),
                 'local_connectivity':list(map(int,local_connectivity)),
                 'n_components':list(map(int, n_components)),
                 'densmap':list(map(bool,densmap))
                 }

    hdbscan_args = {'min_cluster_size':list(map(int,min_cluster_sizes)),
                    'min_samples':list(map(int,min_samples)),
                    'cluster_selection_epsilon':list(map(float,cluster_selection_epsilons)),
                    'cluster_selection_method':cluster_selection_methods}

    obj = umap_hdbscan_grid_sweep(inpath,
                                  outpath,
                                  umap_args,
                                  hdbscan_args)
    obj.run(cores=10)
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
    fire.Fire(run_umap_hdbscan_grid_sweep)
    
#    test_select_hdbscan_clustering()
    #fire.Fire(select_hdbscan_clustering)  
