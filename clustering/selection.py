import pandas as pd
import plotnine as pn
from pathlib import Path
from clustering.fit_tsne import fit_tsne
from visualization.tsne_vis import build_visualization

df = pd.read_csv("/gscratch/comdata/output/reddit_clustering/subreddit_comment_authors-tf_10k_LSI/hdbscan/selection_data.csv",index_col=0)

# plot silhouette_score as a function of isolates
df = df.sort_values("silhouette_score")

df['n_isolates'] = df.n_isolates.str.split("\n0").apply(lambda rg: int(rg[1]))
p = pn.ggplot(df,pn.aes(x='n_isolates',y='silhouette_score')) + pn.geom_point()
p.save("isolates_x_score.png")

p = pn.ggplot(df,pn.aes(y='n_clusters',x='n_isolates',color='silhouette_score')) + pn.geom_point()
p.save("clusters_x_isolates.png")

# the best result for hdbscan seems like this one
best_eom = df[(df.n_isolates <5000)&(df.silhouette_score>0.4)&(df.cluster_selection_method=='eom')&(df.min_cluster_size==2)].iloc[df.shape[1]]

best_lsi = df[(df.n_isolates <5000)&(df.silhouette_score>0.4)&(df.cluster_selection_method=='leaf')&(df.min_cluster_size==2)].iloc[df.shape[1]]

tsne_data = Path("./clustering/authors-tf_lsi850_tsne.feather")

if not tnse_data.exists():
    fit_tsne("/gscratch/comdata/output/reddit_similarity/subreddit_comment_authors-tf_10k_LSI/850.feather",
             tnse_data)

build_visualization("./clustering/authors-tf_lsi850_tsne.feather",
                    Path(best_eom.outpath)/(best_eom['name']+'.feather'),
                    "./authors-tf_lsi850_best_eom.html")

build_visualization("./clustering/authors-tf_lsi850_tsne.feather",
                    Path(best_leaf.outpath)/(best_leaf['name']+'.feather'),
                    "./authors-tf_lsi850_best_leaf.html")

