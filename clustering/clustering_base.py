import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import silhouette_score, silhouette_samples
from collections import Counter

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
        self.cleanup()

    def cleanup(self):
        self.cluster_data = None
        self.mat = None
        self.clustering=None
        self.subreddits=None
        
    def get_info(self):
        if not self.hasrun:
            self.run()

        self.result = clustering_result(outpath=str(self.outpath.resolve()),
                                        silhouette_score=self.score,
                                        name=self.name,
                                        n_clusters=self.n_clusters,
                                        n_isolates=self.n_isolates,
                                        silhouette_samples = self.silsampout
                                        )
        return self.result

    def silhouette(self):
        counts = Counter(self.clustering.labels_)
        singletons = [key for key, value in counts.items() if value == 1]
        isolates = (self.clustering.labels_ == -1) | (np.isin(self.clustering.labels_,np.array(singletons)))
        scoremat = self.mat[~isolates][:,~isolates]
        if self.n_clusters > 1:
            score = silhouette_score(scoremat, self.clustering.labels_[~isolates], metric='precomputed')
            silhouette_samp = silhouette_samples(self.mat, self.clustering.labels_, metric='precomputed')
            silhouette_samp = pd.DataFrame({'subreddit':self.subreddits,'score':silhouette_samp})
            self.outpath.mkdir(parents=True, exist_ok=True)
            silsampout = self.outpath / ("silhouette_samples-" + self.name +  ".feather")
            self.silsampout = silsampout.resolve()
            silhouette_samp.to_feather(self.silsampout)
        else:
            score = None
            self.silsampout = None
        return score

    def read_distance_mat(self, similarities, use_threads=True):
        print(similarities)
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

        n_isolates2 = cluster_sizes.loc[cluster_sizes.cluster==-1,:]['subreddit'].to_list()
        if len(n_isolates2) > 0:
            n_isloates2 = n_isolates2[0]
        print(f"{n_isolates2} subreddits are in cluster -1",flush=True)

        if n_isolates1 == 0:
            self.n_isolates = n_isolates2
        else:
            self.n_isolates = n_isolates1

        return cluster_data

class twoway_clustering_job(clustering_job):
    def __init__(self, infile, outpath, name, call1, call2, args1, args2):
        self.outpath = Path(outpath)
        self.call1 = call1
        self.args1 = args1
        self.call2 = call2
        self.args2 = args2
        self.infile = Path(infile)
        self.name = name
        self.hasrun = False
        self.args = args1|args2

    def run(self):
        self.subreddits, self.mat = self.read_distance_mat(self.infile)
        self.step1 = self.call1(self.mat, **self.args1)
        self.clustering = self.call2(self.mat, self.step1, **self.args2)
        self.cluster_data = self.process_clustering(self.clustering, self.subreddits)
        self.hasrun = True
        self.after_run()
        self.cleanup()

    def after_run():
        self.score = self.silhouette()
        self.outpath.mkdir(parents=True, exist_ok=True)
        print(self.outpath/(self.name+".feather"))
        self.cluster_data.to_feather(self.outpath/(self.name + ".feather"))


    def cleanup(self):
        super().cleanup()
        self.step1 = None

@dataclass
class clustering_result:
    outpath:Path
    silhouette_score:float
    name:str
    n_clusters:int
    n_isolates:int
    silhouette_samples:str
