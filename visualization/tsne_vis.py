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

#    base = base.encode(alt.Color(field='color',type='nominal',scale=alt.Scale(scheme='category10')))

    cluster_dropdown = alt.binding_select(options=[str(c) for c in sorted(set(plot_data.cluster))])

    #    subreddit_dropdown = alt.binding_select(options=sorted(plot_data.subreddit))

    cluster_click_select = alt.selection_single(on='click',fields=['cluster'], bind=cluster_dropdown, name=' ')
    # cluster_select = alt.selection_single(fields=['cluster'], bind=cluster_dropdown, name='cluster')
    # cluster_select_and = cluster_click_select & cluster_select
    #
    #    subreddit_select = alt.selection_single(on='click',fields=['subreddit'],bind=subreddit_dropdown,name='subreddit_click')
    
    base_scale = alt.Scale(scheme={"name":'category10',
                                   "extent":[0,100],
                                   "count":10})

    color = alt.condition(cluster_click_select ,
                          alt.Color(field='color',type='nominal',scale=base_scale),
                          alt.value("lightgray"))
  
    
    base = alt.Chart(plot_data).mark_text().encode(
        alt.X('x',axis=alt.Axis(grid=False),scale=alt.Scale(domain=(-65,65))),
        alt.Y('y',axis=alt.Axis(grid=False),scale=alt.Scale(domain=(-65,65))),
        color=color,
        text='subreddit')

    base = base.add_selection(cluster_click_select)
 

    return base

def zoom_plot(plot_data):
    chart = base_plot(plot_data)

    chart = chart.interactive()
    chart = chart.properties(width=1275,height=800)

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


    sr = sr.properties(width=1275,height=600)


    chart = (viewport1 | viewport2) & sr


    return chart

def assign_cluster_colors(tsne_data, clusters, n_colors, n_neighbors = 4):
    isolate_color = 101

    cluster_sizes = clusters.groupby('cluster').count()
    singletons = set(cluster_sizes.loc[cluster_sizes.subreddit == 1].reset_index().cluster)

    tsne_data = tsne_data.merge(clusters,on='subreddit')
    
    centroids = tsne_data.groupby('cluster').agg({'x':np.mean,'y':np.mean})

    color_ids = np.arange(n_colors)

    distances = np.empty(shape=(centroids.shape[0],centroids.shape[0]))

    groups = tsne_data.groupby('cluster')
    
    points = np.array(tsne_data.loc[:,['x','y']])
    centers = np.array(centroids.loc[:,['x','y']])

    # point x centroid
    point_center_distances = np.linalg.norm((points[:,None,:] - centers[None,:,:]),axis=-1)
    
    # distances is cluster x point
    for gid, group in groups:
        c_dists = point_center_distances[group.index.values,:].min(axis=0)
        distances[group.cluster.values[0],] = c_dists        

    # nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(centroids) 
    # distances, indices = nbrs.kneighbors()

    nearest = distances.argpartition(n_neighbors,0)
    indices = nearest[:n_neighbors,:].T
    # neighbor_distances = np.copy(distances)
    # neighbor_distances.sort(0)
    # neighbor_distances = neighbor_distances[0:n_neighbors,:]
    
    # nbrs = NearestNeighbors(n_neighbors=n_neighbors,metric='precomputed').fit(distances) 
    # distances, indices = nbrs.kneighbors()

    color_assignments = np.repeat(-1,len(centroids))

    for i in range(len(centroids)):
        if (centroids.iloc[i].name == -1) or (i in singletons):
            color_assignments[i] = isolate_color
        else:
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

    # tsne_data = "/gscratch/comdata/output/reddit_tsne/subreddit_author_tf_similarities_10000.feather"
    # clusters = "/gscratch/comdata/output/reddit_clustering/subreddit_author_tf_similarities_10000.feather"

    tsne_data = pd.read_feather(tsne_data)
    tsne_data = tsne_data.rename(columns={'_subreddit':'subreddit'})
    clusters = pd.read_feather(clusters)

    tsne_data = assign_cluster_colors(tsne_data,clusters,10,8)

    sr_per_cluster = tsne_data.groupby('cluster').subreddit.count().reset_index()
    sr_per_cluster = sr_per_cluster.rename(columns={'subreddit':'cluster_size'})

    tsne_data = tsne_data.merge(sr_per_cluster,on='cluster')

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
