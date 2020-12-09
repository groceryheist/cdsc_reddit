import fire
import pyarrow
import pandas as pd
from numpy import random
import numpy as np
from sklearn.manifold import TSNE

similarities = "term_similarities_10000.feather"

def fit_tsne(similarities, output, learning_rate=750, perplexity=50, n_iter=10000, early_exaggeration=20):
    '''
    similarities: feather file with a dataframe of similarity scores
    learning_rate: parameter controlling how fast the model converges. Too low and you get outliers. Too high and you get a ball.
    perplexity: number of neighbors to use. the default of 50 is often good.

    '''
    df = pd.read_feather(similarities)

    n = df.shape[0]
    mat = np.array(df.drop('subreddit',1),dtype=np.float64)
    mat[range(n),range(n)] = 1
    mat[mat > 1] = 1
    dist = 2*np.arccos(mat)/np.pi
    tsne_model = TSNE(2,learning_rate=750,perplexity=50,n_iter=10000,metric='precomputed',early_exaggeration=20,n_jobs=-1)
    tsne_fit_model = tsne_model.fit(dist)

    tsne_fit_whole = tsne_fit_model.fit_transform(dist)

    plot_data = pd.DataFrame({'x':tsne_fit_whole[:,0],'y':tsne_fit_whole[:,1], 'subreddit':df.subreddit})

    plot_data.to_feather(output)

if __name__ == "__main__":
    fire.Fire(fit_tsne)
