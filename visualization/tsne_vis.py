import pyarrow
import altair as alt
alt.data_transformers.disable_max_rows()
alt.data_transformers.enable('data_server')
import pandas as pd
from numpy import random
import numpy as np
from sklearn.manifold import TSNE

df = pd.read_csv("reddit_term_similarity_3000.csv")
df = df.sort_values(['i','j'])

n = max(df.i.max(),df.j.max())

def zero_pad(grp):
    p = grp.shape[0]
    grp = grp.sort_values('j')
    return np.concatenate([np.zeros(n-p),np.zeros(1),np.array(grp.value)])

col_names = df.sort_values('j').loc[:,['subreddit_j']].drop_duplicates()
first_name = list(set(df.subreddit_i) - set(df.subreddit_j))[0]
col_names = [first_name] + list(col_names.subreddit_j)
mat = df.groupby('i').apply(zero_pad)
mat.loc[n] = np.concatenate([np.zeros(n),np.ones(1)])
mat = np.stack(mat)

# plot the matrix using the first and second eigenvalues
mat = mat + np.tril(mat.transpose(),k=-1)

tsne_model = TSNE(2,learning_rate=500,perplexity=40,n_iter=2000)
tsne_fit_model = tsne_model.fit(mat)
tsne_fit_whole = tsne_fit_model.fit_transform(mat)

plot_data = pd.DataFrame({'x':tsne_fit_whole[:,0],'y':tsne_fit_whole[:,1], 'subreddit':col_names})

plot_data.to_feather("tsne_subreddit_fit.feather")

slider = alt.binding_range(min=1,max=100,step=1,name='zoom: ')
selector = alt.selection_single(name='zoomselect',fields=['zoom'],bind='scales',init={'zoom':1})

xrange = plot_data.x.max()-plot_data.x.min()
yrange = plot_data.y.max()-plot_data.y.min()

chart = alt.Chart(plot_data).mark_text().encode(
    alt.X('x',axis=alt.Axis(grid=False)),
    alt.Y('y',axis=alt.Axis(grid=False)),
    text='subreddit')

#chart = chart.add_selection(selector)

chart = chart.configure_view(
    continuousHeight=xrange/20,
    continuousWidth=yrange/20
)

amount_shown = lambda zoom: {'width':xrange/zoom,'height':yrange/zoom}

alt.data_transformers.enable('default')
chart = chart.properties(width=1000,height=1000)
chart = chart.interactive()
chart.save("test_tsne_whole.html")
chart = chart.properties(width=10000,height=10000)
chart.save("test_tsne_whole.svg")
