#srun_cdsc='srun -p comdata-int -A comdata --time=300:00:00 --time-min=00:15:00 --mem=100G --ntasks=1 --cpus-per-task=28'
srun_singularity=source /gscratch/comdata/users/nathante/cdsc_reddit/bin/activate && srun_singularity.sh
similarity_data=/gscratch/comdata/output/reddit_similarity
clustering_data=/gscratch/comdata/output/reddit_clustering
selection_grid="--max_iter=3000 --convergence_iter=15,30,100 --damping=0.5,0.6,0.7,0.8,0.85,0.9,0.95,0.97,0.99, --preference_quantile=0.1,0.3,0.5,0.7,0.9"
#selection_grid="--max_iter=3000 --convergence_iter=[15] --preference_quantile=[0.5] --damping=[0.99]"
all:$(clustering_data)/subreddit_comment_authors_10k/selection_data.csv $(clustering_data)/subreddit_comment_authors-tf_10k/selection_data.csv $(clustering_data)/subreddit_comment_terms_10k/selection_data.csv
# $(clustering_data)/subreddit_comment_authors_30k.feather/SUCCESS $(clustering_data)/subreddit_authors-tf_similarities_30k.feather/SUCCESS
# $(clustering_data)/subreddit_comment_terms_30k.feather/SUCCESS

$(clustering_data)/subreddit_comment_authors_10k/selection_data.csv:selection.py $(similarity_data)/subreddit_comment_authors_10k.feather clustering.py
	$(srun_singularity) python3 selection.py $(similarity_data)/subreddit_comment_authors_10k.feather $(clustering_data)/subreddit_comment_authors_10k $(clustering_data)/subreddit_comment_authors_10k/selection_data.csv $(selection_grid) -J 20

$(clustering_data)/subreddit_comment_terms_10k/selection_data.csv:selection.py $(similarity_data)/subreddit_comment_terms_10k.feather clustering.py
	$(srun_singularity) python3 selection.py $(similarity_data)/subreddit_comment_terms_10k.feather $(clustering_data)/subreddit_comment_terms_10k  $(clustering_data)/subreddit_comment_terms_10k/selection_data.csv $(selection_grid) -J 20 

$(clustering_data)/subreddit_comment_authors-tf_10k/selection_data.csv:clustering.py $(similarity_data)/subreddit_comment_authors-tf_10k.feather
	$(srun_singularity) python3 selection.py $(similarity_data)/subreddit_comment_authors-tf_10k.feather $(clustering_data)/subreddit_comment_authors-tf_10k  $(clustering_data)/subreddit_comment_authors-tf_10k/selection_data.csv $(selection_grid) -J 20

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
