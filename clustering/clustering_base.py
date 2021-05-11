from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import silhouette_score, silhouette_samples
from itertools import product, chain
from multiprocessing import Pool, cpu_count

def sim_to_dist(mat):
    dist = 1-mat
    dist[dist < 0] = 0
    np.fill_diagonal(dist,0)
    return dist

class grid_sweep:
    def __init__(self, jobtype, inpath, outpath, namer, *args):
        self.jobtype = jobtype
        self.namer = namer
        grid = list(product(*args))
        inpath = Path(inpath)
        outpath = Path(outpath)
        self.hasrun = False
        self.grid = [(inpath,outpath,namer(*g)) + g for g in grid]
        self.jobs = [jobtype(*g) for g in self.grid]

    def run(self, cores=20):
        if cores is not None and cores > 1:
            with Pool(cores) as pool:
                infos = pool.map(self.jobtype.get_info, self.jobs)
        else:
            infos = map(self.jobtype.get_info, self.jobs)

        self.infos = pd.DataFrame(infos)
        self.hasrun = True

    def save(self, outcsv):
        if not self.hasrun:
            self.run()
        outcsv = Path(outcsv)
        outcsv.parent.mkdir(parents=True, exist_ok=True)
        self.infos.to_csv(outcsv)


class lsi_grid_sweep(grid_sweep):
    def __init__(self, jobtype, subsweep, inpath, lsi_dimensions, outpath, *args, **kwargs):
        self.jobtype = jobtype
        self.subsweep = subsweep
        inpath = Path(inpath)
        if lsi_dimensions == 'all':
            lsi_paths = list(inpath.glob("*"))
        else:
            lsi_paths = [inpath / (dim + '.feather') for dim in lsi_dimensions]

        lsi_nums = [p.stem for p in lsi_paths]
        self.hasrun = False
        self.subgrids = [self.subsweep(lsi_path, outpath,  lsi_dim, *args, **kwargs) for lsi_dim, lsi_path in zip(lsi_nums, lsi_paths)]
        self.jobs = list(chain(*map(lambda gs: gs.jobs, self.subgrids)))


# this is meant to be an interface, not created directly
class clustering_job:
    def __init__(self, infile, outpath, name, call, *args, **kwargs):
        self.outpath = Path(outpath)
        self.call = call
        self.args = args
        self.kwargs = kwargs
        self.infile = Path(infile)
        self.name = name
        self.hasrun = False

    def run(self):
        self.subreddits, self.mat = self.read_distance_mat(self.infile)
        self.clustering = self.call(self.mat, *self.args, **self.kwargs)
        self.cluster_data = self.process_clustering(self.clustering, self.subreddits)
        self.score = self.silhouette()
        self.outpath.mkdir(parents=True, exist_ok=True)
        self.cluster_data.to_feather(self.outpath/(self.name + ".feather"))
        self.hasrun = True
        
    def get_info(self):
        if not self.hasrun:
            self.run()

        self.result = clustering_result(outpath=str(self.outpath.resolve()),
                                        silhouette_score=self.score,
                                        name=self.name,
                                        n_clusters=self.n_clusters,
                                        n_isolates=self.n_isolates,
                                        silhouette_samples = str(self.silsampout.resolve())
                                        )
        return self.result

    def silhouette(self):
        isolates = self.clustering.labels_ == -1
        scoremat = self.mat[~isolates][:,~isolates]
        score = silhouette_score(scoremat, self.clustering.labels_[~isolates], metric='precomputed')
        silhouette_samp = silhouette_samples(self.mat, self.clustering.labels_, metric='precomputed')
        silhouette_samp = pd.DataFrame({'subreddit':self.subreddits,'score':silhouette_samp})
        self.outpath.mkdir(parents=True, exist_ok=True)
        self.silsampout = self.outpath / ("silhouette_samples-" + self.name +  ".feather")
        silhouette_samp.to_feather(self.silsampout)
        return score

    def read_distance_mat(self, similarities, use_threads=True):
        df = pd.read_feather(similarities, use_threads=use_threads)
        mat = np.array(df.drop('_subreddit',1))
        n = mat.shape[0]
        mat[range(n),range(n)] = 1
        return (df._subreddit,1-mat)

    def process_clustering(self, clustering, subreddits):

        if hasattr(clustering,'n_iter_'):
            print(f"clustering took {clustering.n_iter_} iterations")

        clusters = clustering.labels_
        self.n_clusters = len(set(clusters))

        print(f"found {self.n_clusters} clusters")

        cluster_data = pd.DataFrame({'subreddit': subreddits,'cluster':clustering.labels_})

        cluster_sizes = cluster_data.groupby("cluster").count().reset_index()
        print(f"the largest cluster has {cluster_sizes.loc[cluster_sizes.cluster!=-1].subreddit.max()} members")

        print(f"the median cluster has {cluster_sizes.subreddit.median()} members")
        n_isolates1 = (cluster_sizes.subreddit==1).sum()

        print(f"{n_isolates1} clusters have 1 member")

        n_isolates2 = (cluster_sizes.loc[cluster_sizes.cluster==-1,['subreddit']])

        print(f"{n_isolates2} subreddits are in cluster -1",flush=True)

        if n_isolates1 == 0:
            self.n_isolates = n_isolates2
        else:
            self.n_isolates = n_isolates1

        return cluster_data


class lsi_mixin():
    def set_lsi_dims(self, lsi_dims):
        self.lsi_dims = lsi_dims

@dataclass
class clustering_result:
    outpath:Path
    silhouette_score:float
    name:str
    n_clusters:int
    n_isolates:int
    silhouette_samples:str

@dataclass
class lsi_result_mixin:
    lsi_dimensions:int
