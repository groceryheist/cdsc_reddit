from clustering_base import sim_to_dist, process_clustering_result, clustering_result, read_similarity_mat
from clustering_base import lsi_result_mixin, lsi_mixin, clustering_job, grid_sweep, lsi_grid_sweep
from dataclasses import dataclass
import hdbscan
from sklearn.neighbors import NearestNeighbors
import plotnine as pn
import numpy as np
from itertools import product, starmap, chain
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples
from pathlib import Path
from multiprocessing import Pool, cpu_count
import fire
from pyarrow.feather import write_feather

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

class hdbscan_lsi_grid_sweep(lsi_grid_sweep):
    def __init__(self,
                 inpath,
                 lsi_dims,
                 outpath,
                 min_cluster_sizes,
                 min_samples,
                 cluster_selection_epsilons,
                 cluster_selection_methods
                 ):

        super().__init__(hdbscan_lsi_job,
                         _hdbscan_lsi_grid_sweep,
                         inpath,
                         lsi_dims,
                         outpath,
                         min_cluster_sizes,
                         min_samples,
                         cluster_selection_epsilons,
                         cluster_selection_methods)
        
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


class _hdbscan_lsi_grid_sweep(grid_sweep):
    def __init__(self,
                 inpath,
                 outpath,
                 lsi_dim,
                 *args,
                 **kwargs):

        self.lsi_dim = lsi_dim
        self.jobtype = hdbscan_lsi_job
        super().__init__(self.jobtype, inpath, outpath, self.namer, self.lsi_dim, *args, **kwargs)


    def namer(self, *args, **kwargs):
        s = hdbscan_grid_sweep.namer(self, *args[1:], **kwargs)
        s += f"_lsi-{self.lsi_dim}"
        return s

@dataclass
class hdbscan_clustering_result(clustering_result):
    min_cluster_size:int
    min_samples:int
    cluster_selection_epsilon:float
    cluster_selection_method:str

@dataclass
class hdbscan_clustering_result_lsi(hdbscan_clustering_result, lsi_result_mixin):
    pass 

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

class hdbscan_lsi_job(hdbscan_job, lsi_mixin):
    def __init__(self, infile, outpath, name, lsi_dims, *args, **kwargs):
        super().__init__(
                         infile,
                         outpath,
                         name,
                         *args,
                         **kwargs)
        super().set_lsi_dims(lsi_dims)

    def get_info(self):
        partial_result = super().get_info()
        self.result = hdbscan_clustering_result_lsi(**partial_result.__dict__,
                                                    lsi_dimensions=self.lsi_dims)
        return self.result

# def select_hdbscan_clustering(inpath,
#                               outpath,
#                               outfile=None,
#                               min_cluster_sizes=[2],
#                               min_samples=[1],
#                               cluster_selection_epsilons=[0],
#                               cluster_selection_methods=['eom'],
#                               lsi_dimensions='all'
#                               ):

#     inpath = Path(inpath)
#     outpath = Path(outpath)
#     outpath.mkdir(exist_ok=True, parents=True)
    
#     if lsi_dimensions is None:
#         lsi_paths = [inpath]
#     elif lsi_dimensions == 'all':
#         lsi_paths = list(inpath.glob("*"))

#     else:
#         lsi_paths = [inpath / (dim + '.feather') for dim in lsi_dimensions]

#     if lsi_dimensions is not None:
#         lsi_nums = [p.stem for p in lsi_paths]
#     else:
#         lsi_nums = [None]
#     grid = list(product(lsi_nums,
#                         min_cluster_sizes,
#                         min_samples,
#                         cluster_selection_epsilons,
#                         cluster_selection_methods))

#     # fix the output file names
#     names = list(map(lambda t:'_'.join(map(str,t)),grid))

#     grid = [(inpath/(str(t[0])+'.feather'),outpath/(name + '.feather'), t[0], name) + t[1:] for t, name in zip(grid, names)]
        
#     with Pool(int(cpu_count()/4)) as pool:
#         mods = starmap(hdbscan_clustering, grid)

#     res = pd.DataFrame(mods)
#     if outfile is None:
#         outfile = outpath / "selection_data.csv"

#     res.to_csv(outfile)

# def hdbscan_clustering(similarities, output, lsi_dim, name, min_cluster_size=2, min_samples=1, cluster_selection_epsilon=0, cluster_selection_method='eom'):
#     subreddits, mat = read_similarity_mat(similarities)
#     mat = sim_to_dist(mat)
#     clustering = _hdbscan_clustering(mat,
#                                      min_cluster_size=min_cluster_size,
#                                      min_samples=min_samples,
#                                      cluster_selection_epsilon=cluster_selection_epsilon,
#                                      cluster_selection_method=cluster_selection_method,
#                                      metric='precomputed',
#                                      core_dist_n_jobs=cpu_count()
#                                      )

#     cluster_data = process_clustering_result(clustering, subreddits)
#     isolates = clustering.labels_ == -1
#     scoremat = mat[~isolates][:,~isolates]
#     score = silhouette_score(scoremat, clustering.labels_[~isolates], metric='precomputed')
#     cluster_data.to_feather(output)
#     silhouette_samp = silhouette_samples(mat, clustering.labels_, metric='precomputed')
#     silhouette_samp = pd.DataFrame({'subreddit':subreddits,'score':silhouette_samp})
#     silsampout = output.parent / ("silhouette_samples" + output.name)
#     silhouette_samp.to_feather(silsampout)

#     result = hdbscan_clustering_result(outpath=output,
#                                        silhouette_samples=silsampout,
#                                        silhouette_score=score,
#                                        name=name,
#                                        min_cluster_size=min_cluster_size,
#                                        min_samples=min_samples,
#                                        cluster_selection_epsilon=cluster_selection_epsilon,
#                                        cluster_selection_method=cluster_selection_method,
#                                        lsi_dimensions=lsi_dim,
#                                        n_isolates=isolates.sum(),
#                                        n_clusters=len(set(clustering.labels_))
#                                    )


                                       
#     return(result)

# # for all runs we should try cluster_selection_epsilon = None
# # for terms we should try cluster_selection_epsilon around 0.56-0.66
# # for authors we should try cluster_selection_epsilon around 0.98-0.99
# def _hdbscan_clustering(mat, *args, **kwargs):
#     print(f"running hdbscan clustering. args:{args}. kwargs:{kwargs}")

#     print(mat)
#     clusterer = hdbscan.HDBSCAN(*args,
#                                 **kwargs,
#                                 )
    
#     clustering = clusterer.fit(mat.astype('double'))
    
#     return(clustering)

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
    fire.Fire{'grid_sweep':hdbscan_grid_sweep,
              'grid_sweep_lsi':hdbscan_lsi_grid_sweep
              'cluster':hdbscan_job,
              'cluster_lsi':hdbscan_lsi_job}
    
#    test_select_hdbscan_clustering()
    #fire.Fire(select_hdbscan_clustering)  
