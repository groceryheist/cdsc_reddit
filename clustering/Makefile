srun_singularity=srun -p compute-bigmem -A comdata --time=48:00:00 --mem=362G -c 40 /bin/bash -c 
similarity_data=../../data/reddit_similarity
clustering_data=../../data/reddit_clustering
kmeans_selection_grid=--max_iters=[3000] --n_inits=[10] --n_clusters=[100,500,1000,1250,1500,1750,2000]
hdbscan_selection_grid=--min_cluster_sizes=[2,3,4,5] --min_samples=[2,3,4,5] --cluster_selection_epsilons=[0,0.01,0.05,0.1,0.15,0.2] --cluster_selection_methods=[eom,leaf]
affinity_selection_grid=--dampings=[0.5,0.6,0.7,0.8,0.95,0.97,0.99] --preference_quantiles=[0.1,0.3,0.5,0.7,0.9] --convergence_iters=[15]

authors_tf_10k_input_lsi=$(similarity_data)/subreddit_comment_authors-tf_10k_LSI
authors_tf_10k_output_lsi=$(clustering_data)/subreddit_comment_authors-tf_10k_LSI

all:authors_tf_10k_lsi

authors_tf_10k_lsi:${authors_tf_10k_output_lsi}/kmeans/selection_data.csv ${authors_tf_10k_output_lsi}/hdbscan/selection_data.csv ${authors_tf_10k_output_lsi}/affinity/selection_data.csv

## LSI Models
${authors_tf_10k_output_lsi}/kmeans/selection_data.csv:clustering.py ${authors_tf_10k_input_lsi} clustering_base.py kmeans_clustering.py
	$(srun_singularity) -c "source ~/.bashrc; python3 kmeans_clustering_lsi.py --inpath=${authors_tf_10k_input_lsi} --outpath=${authors_tf_10k_output_lsi}/kmeans --savefile=${authors_tf_10k_output_lsi}/kmeans/selection_data.csv $(kmeans_selection_grid)"

${authors_tf_10k_output_lsi}/affinity/selection_data.csv:clustering.py ${authors_tf_10k_input_lsi} clustering_base.py affinity_clustering.py
	$(srun_singularity) -c "source ~/.bashrc; python3 affinity_clustering_lsi.py --inpath=${authors_tf_10k_input_lsi} --outpath=${authors_tf_10k_output_lsi}/affinity --savefile=${authors_tf_10k_output_lsi}/affinity/selection_data.csv $(affinity_selection_grid)"

${authors_tf_10k_output_lsi}/hdbscan/selection_data.csv:clustering.py ${authors_tf_10k_input_lsi} clustering_base.py hdbscan_clustering.py
	$(srun_singularity) -c "source ~/.bashrc; python3 hdbscan_clustering_lsi.py --inpath=${authors_tf_10k_input_lsi} --outpath=${authors_tf_10k_output_lsi}/hdbscan --savefile=${authors_tf_10k_output_lsi}/hdbscan/selection_data.csv $(hdbscan_selection_grid)"

${authors_tf_10k_output_lsi}/best_hdbscan.feather:${authors_tf_10k_output_lsi}/hdbscan/selection_data.csv pick_best_clustering.py
	$(srun_singularity) -c "source ~/.bashrc; python3 pick_best_clustering.py $< $@ --min_clusters=50 --max_isolates=5000 --min_cluster_size=2"

${authors_tf_10k_input_lsi}:
	$(MAKE) -C ../similarity

clean:
	rm -f ${authors_tf_10k_output_lsi}/affinity/selection_data.csv
	rm -f ${authors_tf_10k_output_lsi}/kmeans/selection_data.csv
	rm -f ${authors_tf_10k_output_lsi}/hdbscan/selection_data.csv

PHONY: clean 
