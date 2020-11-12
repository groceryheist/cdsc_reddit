import pyarrow
import altair as alt
alt.data_transformers.disable_max_rows()
alt.data_transformers.enable('data_server')
import pandas as pd
from numpy import random
import numpy as np
from sklearn.manifold import TSNE

pd.read_feather("tsne_subreddit_fit.feather")

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
