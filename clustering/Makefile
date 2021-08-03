#srun_cdsc='srun -p comdata-int -A comdata --time=300:00:00 --time-min=00:15:00 --mem=100G --ntasks=1 --cpus-per-task=28'
srun_singularity=source /gscratch/comdata/users/nathante/cdsc_reddit/bin/activate && srun_singularity.sh
similarity_data=/gscratch/comdata/output/reddit_similarity
clustering_data=/gscratch/comdata/output/reddit_clustering
kmeans_selection_grid=--max_iters=[3000] --n_inits=[10] --n_clusters=[100,500,1000,1250,1500,1750,2000]
hdbscan_selection_grid=--min_cluster_sizes=[2,3,4,5] --min_samples=[2,3,4,5] --cluster_selection_epsilons=[0,0.01,0.05,0.1,0.15,0.2] --cluster_selection_methods=[eom,leaf]
affinity_selection_grid=--dampings=[0.5,0.6,0.7,0.8,0.95,0.97,0.99] --preference_quantiles=[0.1,0.3,0.5,0.7,0.9] --convergence_iters=[15]

authors_10k_input=$(similarity_data)/subreddit_comment_authors_10k.feather
authors_10k_input_lsi=$(similarity_data)/subreddit_comment_authors_10k_LSI
authors_10k_output=$(clustering_data)/subreddit_comment_authors_10k
authors_10k_output_lsi=$(clustering_data)/subreddit_comment_authors_10k_LSI

authors_tf_10k_input=$(similarity_data)/subreddit_comment_authors-tf_10k.feather
authors_tf_10k_input_lsi=$(similarity_data)/subreddit_comment_authors-tf_10k_LSI
authors_tf_10k_output=$(clustering_data)/subreddit_comment_authors-tf_10k
authors_tf_10k_output_lsi=$(clustering_data)/subreddit_comment_authors-tf_10k_LSI

terms_10k_input=$(similarity_data)/subreddit_comment_terms_10k.feather
terms_10k_input_lsi=$(similarity_data)/subreddit_comment_terms_10k_LSI
terms_10k_output=$(clustering_data)/subreddit_comment_terms_10k
terms_10k_output_lsi=$(clustering_data)/subreddit_comment_terms_10k_LSI

all:terms_10k authors_10k authors_tf_10k terms_10k_lsi authors_10k_lsi authors_tf_10k_lsi

terms_10k:${terms_10k_output}/kmeans/selection_data.csv ${terms_10k_output}/affinity/selection_data.csv ${terms_10k_output}/hdbscan/selection_data.csv

authors_10k:${authors_10k_output}/kmeans/selection_data.csv ${authors_10k_output}/hdbscan/selection_data.csv ${authors_10k_output}/affinity/selection_data.csv

authors_tf_10k:${authors_tf_10k_output}/kmeans/selection_data.csv ${authors_tf_10k_output}/hdbscan/selection_data.csv ${authors_tf_10k_output}/affinity/selection_data.csv

terms_10k_lsi:${terms_10k_output_lsi}/kmeans/selection_data.csv ${terms_10k_output_lsi}/affinity/selection_data.csv ${terms_10k_output_lsi}/hdbscan/selection_data.csv

authors_10k_lsi:${authors_10k_output_lsi}/kmeans/selection_data.csv ${authors_10k_output_lsi}/hdbscan/selection_data.csv ${authors_10k_output_lsi}/affinity/selection_data.csv

authors_tf_10k_lsi:${authors_tf_10k_output_lsi}/kmeans/selection_data.csv ${authors_tf_10k_output_lsi}/hdbscan/selection_data.csv ${authors_tf_10k_output_lsi}/affinity/selection_data.csv

${authors_10k_output}/kmeans/selection_data.csv:selection.py ${authors_10k_input} clustering_base.py kmeans_clustering.py
	$(srun_singularity) python3 kmeans_clustering.py --inpath=${authors_10k_input} --outpath=${authors_10k_output}/kmeans --savefile=${authors_10k_output}/kmeans/selection_data.csv $(kmeans_selection_grid) 

${terms_10k_output}/kmeans/selection_data.csv:selection.py ${terms_10k_input} clustering_base.py kmeans_clustering.py
	$(srun_singularity) python3 kmeans_clustering.py --inpath=${terms_10k_input} --outpath=${terms_10k_output}/kmeans  --savefile=${terms_10k_output}/kmeans/selection_data.csv $(kmeans_selection_grid) 

${authors_tf_10k_output}/kmeans/selection_data.csv:clustering.py ${authors_tf_10k_input} clustering_base.py kmeans_clustering.py
	$(srun_singularity) python3 kmeans_clustering.py --inpath=${authors_tf_10k_input} --outpath=${authors_tf_10k_output}/kmeans --savefile=${authors_tf_10k_output}/kmeans/selection_data.csv $(kmeans_selection_grid) 

${authors_10k_output}/affinity/selection_data.csv:selection.py ${authors_10k_input} clustering_base.py affinity_clustering.py
	$(srun_singularity) python3 affinity_clustering.py --inpath=${authors_10k_input} --outpath=${authors_10k_output}/affinity --savefile=${authors_10k_output}/affinity/selection_data.csv $(affinity_selection_grid) 

${terms_10k_output}/affinity/selection_data.csv:selection.py ${terms_10k_input} clustering_base.py affinity_clustering.py
	$(srun_singularity) python3 affinity_clustering.py --inpath=${terms_10k_input} --outpath=${terms_10k_output}/affinity  --savefile=${terms_10k_output}/affinity/selection_data.csv $(affinity_selection_grid) 

${authors_tf_10k_output}/affinity/selection_data.csv:clustering.py ${authors_tf_10k_input} clustering_base.py affinity_clustering.py
	$(srun_singularity) python3 affinity_clustering.py --inpath=${authors_tf_10k_input} --outpath=${authors_tf_10k_output}/affinity --savefile=${authors_tf_10k_output}/affinity/selection_data.csv $(affinity_selection_grid) 

${authors_10k_output}/hdbscan/selection_data.csv:selection.py ${authors_10k_input} clustering_base.py hdbscan_clustering.py
	$(srun_singularity) python3 hdbscan_clustering.py --inpath=${authors_10k_input} --outpath=${authors_10k_output}/hdbscan --savefile=${authors_10k_output}/hdbscan/selection_data.csv $(hdbscan_selection_grid) 

${terms_10k_output}/hdbscan/selection_data.csv:selection.py ${terms_10k_input} clustering_base.py hdbscan_clustering.py
	$(srun_singularity) python3 hdbscan_clustering.py --inpath=${terms_10k_input} --outpath=${terms_10k_output}/hdbscan  --savefile=${terms_10k_output}/hdbscan/selection_data.csv $(hdbscan_selection_grid) 

${authors_tf_10k_output}/hdbscan/selection_data.csv:clustering.py ${authors_tf_10k_input} clustering_base.py hdbscan_clustering.py
	$(srun_singularity) python3 hdbscan_clustering.py --inpath=${authors_tf_10k_input} --outpath=${authors_tf_10k_output}/hdbscan --savefile=${authors_tf_10k_output}/hdbscan/selection_data.csv $(hdbscan_selection_grid) 


## LSI Models
${authors_10k_output_lsi}/kmeans/selection_data.csv:selection.py ${authors_10k_input_lsi} clustering_base.py kmeans_clustering.py
	$(srun_singularity) python3 kmeans_clustering_lsi.py --inpath=${authors_10k_input_lsi} --outpath=${authors_10k_output_lsi}/kmeans --savefile=${authors_10k_output_lsi}/kmeans/selection_data.csv $(kmeans_selection_grid)

${terms_10k_output_lsi}/kmeans/selection_data.csv:selection.py ${terms_10k_input_lsi} clustering_base.py kmeans_clustering.py
	$(srun_singularity) python3 kmeans_clustering_lsi.py --inpath=${terms_10k_input_lsi} --outpath=${terms_10k_output_lsi}/kmeans  --savefile=${terms_10k_output_lsi}/kmeans/selection_data.csv $(kmeans_selection_grid)

${authors_tf_10k_output_lsi}/kmeans/selection_data.csv:clustering.py ${authors_tf_10k_input_lsi} clustering_base.py kmeans_clustering.py
	$(srun_singularity) python3 kmeans_clustering_lsi.py --inpath=${authors_tf_10k_input_lsi} --outpath=${authors_tf_10k_output_lsi}/kmeans --savefile=${authors_tf_10k_output_lsi}/kmeans/selection_data.csv $(kmeans_selection_grid)

${authors_10k_output_lsi}/affinity/selection_data.csv:selection.py ${authors_10k_input_lsi} clustering_base.py affinity_clustering.py
	$(srun_singularity) python3 affinity_clustering_lsi.py --inpath=${authors_10k_input_lsi} --outpath=${authors_10k_output_lsi}/affinity --savefile=${authors_10k_output_lsi}/affinity/selection_data.csv $(affinity_selection_grid)

${terms_10k_output_lsi}/affinity/selection_data.csv:selection.py ${terms_10k_input_lsi} clustering_base.py affinity_clustering.py
	$(srun_singularity) python3 affinity_clustering_lsi.py --inpath=${terms_10k_input_lsi} --outpath=${terms_10k_output_lsi}/affinity  --savefile=${terms_10k_output_lsi}/affinity/selection_data.csv $(affinity_selection_grid)

${authors_tf_10k_output_lsi}/affinity/selection_data.csv:clustering.py ${authors_tf_10k_input_lsi} clustering_base.py affinity_clustering.py
	$(srun_singularity) python3 affinity_clustering_lsi.py --inpath=${authors_tf_10k_input_lsi} --outpath=${authors_tf_10k_output_lsi}/affinity --savefile=${authors_tf_10k_output_lsi}/affinity/selection_data.csv $(affinity_selection_grid)

${authors_10k_output_lsi}/hdbscan/selection_data.csv:selection.py ${authors_10k_input_lsi} clustering_base.py hdbscan_clustering.py
	$(srun_singularity) python3 hdbscan_clustering_lsi.py --inpath=${authors_10k_input_lsi} --outpath=${authors_10k_output_lsi}/hdbscan --savefile=${authors_10k_output_lsi}/hdbscan/selection_data.csv $(hdbscan_selection_grid)

${terms_10k_output_lsi}/hdbscan/selection_data.csv:selection.py ${terms_10k_input_lsi} clustering_base.py hdbscan_clustering.py
	$(srun_singularity) python3 hdbscan_clustering_lsi.py --inpath=${terms_10k_input_lsi} --outpath=${terms_10k_output_lsi}/hdbscan  --savefile=${terms_10k_output_lsi}/hdbscan/selection_data.csv $(hdbscan_selection_grid)

${authors_tf_10k_output_lsi}/hdbscan/selection_data.csv:clustering.py ${authors_tf_10k_input_lsi} clustering_base.py hdbscan_clustering.py
	$(srun_singularity) python3 hdbscan_clustering_lsi.py --inpath=${authors_tf_10k_input_lsi} --outpath=${authors_tf_10k_output_lsi}/hdbscan --savefile=${authors_tf_10k_output_lsi}/hdbscan/selection_data.csv $(hdbscan_selection_grid)

${terms_10k_output_lsi}/best_hdbscan.feather:${terms_10k_output_lsi}/hdbscan/selection_data.csv pick_best_clustering.py
	$(srun_singularity) python3 pick_best_clustering.py $< $@ --min_clusters=50 --max_isolates=5000 --min_cluster_size=2

${authors_tf_10k_output_lsi}/best_hdbscan.feather:${authors_tf_10k_output_lsi}/hdbscan/selection_data.csv pick_best_clustering.py
	$(srun_singularity) python3 pick_best_clustering.py $< $@ --min_clusters=50 --max_isolates=5000 --min_cluster_size=2

clean_affinity:
	rm -f ${authors_10k_output}/affinity/selection_data.csv
	rm -f ${authors_tf_10k_output}/affinity/selection_data.csv
	rm -f ${terms_10k_output}/affinity/selection_data.csv

clean_kmeans:
	rm -f ${authors_10k_output}/kmeans/selection_data.csv
	rm -f ${authors_tf_10k_output}/kmeans/selection_data.csv
	rm -f ${terms_10k_output}/kmeans/selection_data.csv

clean_hdbscan:
	rm -f ${authors_10k_output}/hdbscan/selection_data.csv
	rm -f ${authors_tf_10k_output}/hdbscan/selection_data.csv
	rm -f ${terms_10k_output}/hdbscan/selection_data.csv

clean_authors:
	rm -f ${authors_10k_output}/affinity/selection_data.csv
	rm -f ${authors_10k_output}/kmeans/selection_data.csv
	rm -f ${authors_10k_output}/hdbscan/selection_data.csv

clean_authors_tf:
	rm -f ${authors_tf_10k_output}/affinity/selection_data.csv
	rm -f ${authors_tf_10k_output}/kmeans/selection_data.csv
	rm -f ${authors_tf_10k_output}/hdbscan/selection_data.csv

clean_terms:
	rm -f ${terms_10k_output}/affinity/selection_data.csv
	rm -f ${terms_10k_output}/kmeans/selection_data.csv
	rm -f ${terms_10k_output}/hdbscan/selection_data.csv

clean_lsi_affinity:
	rm -f ${authors_10k_output_lsi}/affinity/selection_data.csv
	rm -f ${authors_tf_10k_output_lsi}/affinity/selection_data.csv
	rm -f ${terms_10k_output_lsi}/affinity/selection_data.csv

clean_lsi_kmeans:
	rm -f ${authors_10k_output_lsi}/kmeans/selection_data.csv
	rm -f ${authors_tf_10k_output_lsi}/kmeans/selection_data.csv
	rm -f ${terms_10k_output_lsi}/kmeans/selection_data.csv

clean_lsi_hdbscan:
	rm -f ${authors_10k_output_lsi}/hdbscan/selection_data.csv
	rm -f ${authors_tf_10k_output_lsi}/hdbscan/selection_data.csv
	rm -f ${terms_10k_output_lsi}/hdbscan/selection_data.csv

clean_lsi_authors:
	rm -f ${authors_10k_output_lsi}/affinity/selection_data.csv
	rm -f ${authors_10k_output_lsi}/kmeans/selection_data.csv
	rm -f ${authors_10k_output_lsi}/hdbscan/selection_data.csv

clean_lsi_authors_tf:
	rm -f ${authors_tf_10k_output_lsi}/affinity/selection_data.csv
	rm -f ${authors_tf_10k_output_lsi}/kmeans/selection_data.csv
	rm -f ${authors_tf_10k_output_lsi}/hdbscan/selection_data.csv

clean_lsi_terms:
	rm -f ${terms_10k_output_lsi}/affinity/selection_data.csv
	rm -f ${terms_10k_output_lsi}/kmeans/selection_data.csv
	rm -f ${terms_10k_output_lsi}/hdbscan/selection_data.csv

clean: clean_affinity clean_kmeans clean_hdbscan

PHONY: clean clean_affinity clean_kmeans clean_hdbscan clean_authors clean_authors_tf clean_terms terms_10k authors_10k authors_tf_10k

# $(clustering_data)/subreddit_comment_authors_30k.feather/SUCCESS:selection.py $(similarity_data)/subreddit_comment_authors_30k.feather clustering.py
# 	$(srun_singularity) python3 selection.py $(similarity_data)/subreddit_comment_authors_30k.feather $(clustering_data)/subreddit_comment_authors_30k $(selection_grid) -J 10 && touch $(clustering_data)/subreddit_comment_authors_30k.feather/SUCCESS

# $(clustering_data)/subreddit_comment_terms_30k.feather/SUCCESS:selection.py $(similarity_data)/subreddit_comment_terms_30k.feather clustering.py
# 	$(srun_singularity) python3 selection.py $(similarity_data)/subreddit_comment_terms_30k.feather $(clustering_data)/subreddit_comment_terms_30k $(selection_grid) -J 10 && touch $(clustering_data)/subreddit_comment_terms_30k.feather/SUCCESS

# $(clustering_data)/subreddit_authors-tf_similarities_30k.feather/SUCCESS:clustering.py $(similarity_data)/subreddit_comment_authors-tf_30k.feather
# 	$(srun_singularity) python3 selection.py $(similarity_data)/subreddit_comment_authors-tf_30k.feather $(clustering_data)/subreddit_comment_authors-tf_30k $(selection_grid) -J 8 && touch $(clustering_data)/subreddit_authors-tf_similarities_30k.feather/SUCCESS


# $(clustering_data)/subreddit_comment_authors_100k.feather:clustering.py $(similarity_data)/subreddit_comment_authors_100k.feather
# 	 $(srun_singularity) python3 clustering.py $(similarity_data)/subreddit_comment_authors_100k.feather $(clustering_data)/subreddit_comment_authors_100k.feather ---max_iter=400 --convergence_iter=15 --preference_quantile=0.85 --damping=0.85

# $(clustering_data)/comment_terms_100k.feather:clustering.py $(similarity_data)/subreddit_comment_terms_100k.feather
# 	$(srun_singularity) python3 clustering.py $(similarity_data)/comment_terms_10000.feather $(clustering_data)/comment_terms_10000.feather ---max_iter=1000 --convergence_iter=15 --preference_quantile=0.9 --damping=0.5

# $(clustering_data)/subreddit_comment_author-tf_100k.feather:clustering.py $(similarity_data)/subreddit_comment_author-tf_100k.feather
# 	$(srun_singularity) python3 clustering.py $(similarity_data)/subreddit_comment_author-tf_100k.parquet $(clustering_data)/subreddit_comment_author-tf_100k.feather ---max_iter=400 --convergence_iter=15 --preference_quantile=0.5 --damping=0.85


# it's pretty difficult to get a result that isn't one huge megacluster. A sign that it's bullcrap
# /gscratch/comdata/output/reddit_clustering/wang_similarity_10000.feather:clustering.py /gscratch/comdata/output/reddit_similarity/wang_similarity_10000.feather
# 	./clustering.py /gscratch/comdata/output/reddit_similarity/wang_similarity_10000.feather /gscratch/comdata/output/reddit_clustering/wang_similarity_10000.feather ---max_iter=400 --convergence_iter=15 --preference_quantile=0.9 --damping=0.85

# /gscratch/comdata/output/reddit_tsne/subreddit_author_tf_similarities_10000.feather:fit_tsne.py /gscratch/comdata/output/reddit_similarity/subreddit_author_tf_similarities_10000.parquet

# 	start_spark_and_run.sh 1 fit_tsne.py --similarities=/gscratch/comdata/output/reddit_similarity/subreddit_author_tf_similarities_10000.parquet --output=/gscratch/comdata/output/reddit_tsne/subreddit_author_tf_similarities_10000.feather


# /gscratch/comdata/output/reddit_tsne/wang_similarity_10000.feather:fit_tsne.py /gscratch/comdata/output/reddit_similarity/wang_similarity_10000.feather

# 	python3 fit_tsne.py --similarities=/gscratch/comdata/output/reddit_similarity/wang_similarity_10000.feather --output=/gscratch/comdata/output/reddit_tsne/wang_similarity_10000.feather

# /gscratch/comdata/output/reddit_tsne/comment_authors_10000.feather:clustering.py /gscratch/comdata/output/reddit_similarity/comment_authors_10000.feather
# #	$srun_cdsc python3
# 	start_spark_and_run.sh 1 fit_tsne.py --similarities=/gscratch/comdata/output/reddit_similarity/comment_authors_10000.feather --output=/gscratch/comdata/output/reddit_tsne/comment_authors_10000.feather
