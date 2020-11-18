import pyarrow
import altair as alt
alt.data_transformers.disable_max_rows()
alt.data_transformers.enable('default')
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from numpy import random
import fire
import numpy as np

def base_plot(plot_data):
    base = alt.Chart(plot_data).mark_text().encode(
        alt.X('x',axis=alt.Axis(grid=False),scale=alt.Scale(domain=(-65,65))),
        alt.Y('y',axis=alt.Axis(grid=False),scale=alt.Scale(domain=(-65,65))),
        text='subreddit')

    return base

def zoom_plot(plot_data):
    chart = base_plot(plot_data)
    chart = chart.encode(alt.Color(field='color',type='nominal',scale=alt.Scale(scheme='category10')))
    chart = chart.interactive()
    chart = chart.properties(width=1275,height=1000)

    return chart

def viewport_plot(plot_data):
    selector1 = alt.selection_interval(encodings=['x','y'],init={'x':(-65,65),'y':(-65,65)})
    selectorx2 = alt.selection_interval(encodings=['x'],init={'x':(30,40)})
    selectory2 = alt.selection_interval(encodings=['y'],init={'y':(-20,0)})

    base = base_plot(plot_data)

    viewport = base.mark_point(fillOpacity=0.2,opacity=0.2).encode(
        alt.X('x',axis=alt.Axis(grid=False)),
        alt.Y('y',axis=alt.Axis(grid=False)),
    )

    viewport = viewport.properties(width=600,height=400)

    viewport1 = viewport.add_selection(selector1)

    viewport2 = viewport.encode(
        alt.X('x',axis=alt.Axis(grid=False),scale=alt.Scale(domain=selector1)),
        alt.Y('y',axis=alt.Axis(grid=False),scale=alt.Scale(domain=selector1))
    )

    viewport2 = viewport2.add_selection(selectorx2)
    viewport2 = viewport2.add_selection(selectory2)

    sr = base.encode(alt.X('x',axis=alt.Axis(grid=False),scale=alt.Scale(domain=selectorx2)),
                     alt.Y('y',axis=alt.Axis(grid=False),scale=alt.Scale(domain=selectory2))
    )

    sr = sr.encode(alt.Color(field='color',type='nominal',scale=alt.Scale(scheme='category10')))
    sr = sr.properties(width=1275,height=600)


    chart = (viewport1 | viewport2) & sr


    return chart

def assign_cluster_colors(tsne_data, clusters, n_colors, n_neighbors = 4):
    tsne_data = tsne_data.merge(clusters,on='subreddit')
    
    centroids = tsne_data.groupby('cluster').agg({'x':np.mean,'y':np.mean})

    color_ids = np.arange(n_colors)

    distances = np.empty(shape=(centroids.shape[0],centroids.shape[0]))

    groups = tsne_data.groupby('cluster')
    for centroid in centroids.itertuples():
        c_dists = groups.apply(lambda r: min(np.sqrt(np.square(centroid.x - r.x) + np.square(centroid.y-r.y))))
        distances[:,centroid.Index] = c_dists

    # nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(centroids) 
    # distances, indices = nbrs.kneighbors()

    nbrs = NearestNeighbors(n_neighbors=n_neighbors,metric='precomputed').fit(distances) 
    distances, indices = nbrs.kneighbors()

    color_assignments = np.repeat(-1,len(centroids))

    for i in range(len(centroids)):
        knn = indices[i]
        knn_colors = color_assignments[knn]
        available_colors = color_ids[list(set(color_ids) - set(knn_colors))]

        if(len(available_colors) > 0):
            color_assignments[i] = available_colors[0]
        else:
            raise Exception("Can't color this many neighbors with this many colors")


    centroids = centroids.reset_index()
    colors = centroids.loc[:,['cluster']]
    colors['color'] = color_assignments

    tsne_data = tsne_data.merge(colors,on='cluster')
    return(tsne_data)

def build_visualization(tsne_data, clusters, output):

    tsne_data = pd.read_feather(tsne_data)
    clusters = pd.read_feather(clusters)

    tsne_data = assign_cluster_colors(tsne_data,clusters,10,8)

    term_zoom_plot = zoom_plot(tsne_data)

    term_zoom_plot.save(output)

    term_viewport_plot = viewport_plot(tsne_data)

    term_viewport_plot.save(output.replace(".html","_viewport.html"))

if __name__ == "__main__":
    fire.Fire(build_visualization)

# commenter_data = pd.read_feather("tsne_author_fit.feather")
# clusters = pd.read_feather('author_3000_clusters.feather')
# commenter_data = assign_cluster_colors(commenter_data,clusters,10,8)
# commenter_zoom_plot = zoom_plot(commenter_data)
# commenter_viewport_plot = viewport_plot(commenter_data)
# commenter_zoom_plot.save("subreddit_commenters_tsne_3000.html")
# commenter_viewport_plot.save("subreddit_commenters_tsne_3000_viewport.html")

# chart = chart.properties(width=10000,height=10000)
# chart.save("test_tsne_whole.svg")
