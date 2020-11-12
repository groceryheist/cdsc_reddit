import pyarrow
import pandas as pd
from numpy import random
import numpy as np
from sklearn.manifold import TSNE

df = pd.read_feather("reddit_term_similarity_3000.feather")
df = df.sort_values(['i','j'])

n = max(df.i.max(),df.j.max())

def zero_pad(grp):
    p = grp.shape[0]
    grp = grp.sort_values('j')
    return np.concatenate([np.zeros(n-p),np.ones(1),np.array(grp.value)])

col_names = df.sort_values('j').loc[:,['subreddit_j']].drop_duplicates()
first_name = list(set(df.subreddit_i) - set(df.subreddit_j))[0]
col_names = [first_name] + list(col_names.subreddit_j)
mat = df.groupby('i').apply(zero_pad)
mat.loc[n] = np.concatenate([np.zeros(n),np.ones(1)])
mat = np.stack(mat)

mat = mat + np.tril(mat.transpose(),k=-1)
dist = 2*np.arccos(mat)/np.pi

tsne_model = TSNE(2,learning_rate=500,perplexity=50,n_iter=10000,metric='precomputed',early_exaggeration=20,n_jobs=-1)

tsne_fit_model = tsne_model.fit(dist)

tsne_fit_whole = tsne_fit_model.fit_transform(dist)

plot_data = pd.DataFrame({'x':tsne_fit_whole[:,0],'y':tsne_fit_whole[:,1], 'subreddit':col_names})

plot_data.to_feather("tsne_subreddit_fit.feather")
