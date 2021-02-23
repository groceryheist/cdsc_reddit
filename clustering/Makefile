#srun_cdsc='srun -p comdata-int -A comdata --time=300:00:00 --time-min=00:15:00 --mem=100G --ntasks=1 --cpus-per-task=28'
all:/gscratch/comdata/output/reddit_clustering/comment_authors_10000.feather /gscratch/comdata/output/reddit_clustering/comment_terms_10000.feather /gscratch/comdata/output/reddit_clustering/subreddit_author_tf_similarities_10000.feather /gscratch/comdata/output/reddit_tsne/subreddit_author_tf_similarities_10000.feather /gscratch/comdata/output/reddit_tsne/comment_authors_10000.feather
#all:/gscratch/comdata/output/reddit_clustering/comment_authors_10000.feather /gscratch/comdata/output/reddit_clustering/comment_terms_10000.feather /gscratch/comdata/output/reddit_clustering/subreddit_author_tf_similarities_10000.feather /gscratch/comdata/output/reddit_clustering/wang_similarity_10000.feather /gscratch/comdata/output/reddit_tsne/subreddit_author_tf_similarities_10000.feather /gscratch/comdata/output/reddit_tsne/wang_similarity_10000.feather /gscratch/comdata/output/reddit_tsne/comment_authors_10000.feather

/gscratch/comdata/output/reddit_clustering/comment_authors_10000.feather:clustering.py /gscratch/comdata/output/reddit_similarity/comment_authors_10000.feather
#	$srun_cdsc python3
	start_spark_and_run.sh 1 clustering.py /gscratch/comdata/output/reddit_similarity/comment_authors_10000.feather /gscratch/comdata/output/reddit_clustering/comment_authors_10000.feather ---max_iter=400 --convergence_iter=15 --preference_quantile=0.85 --damping=0.85

/gscratch/comdata/output/reddit_clustering/comment_terms_10000.feather:clustering.py /gscratch/comdata/output/reddit_similarity/comment_terms_10000.feather
#	$srun_cdsc python3
	start_spark_and_run.sh 1 clustering.py /gscratch/comdata/output/reddit_similarity/comment_terms_10000.feather /gscratch/comdata/output/reddit_clustering/comment_terms_10000.feather ---max_iter=1000 --convergence_iter=15 --preference_quantile=0.9 --damping=0.5

/gscratch/comdata/output/reddit_clustering/subreddit_author_tf_similarities_10000.feather:clustering.py /gscratch/comdata/output/reddit_similarity/subreddit_author_tf_similarities_10000.parquet
#	$srun_cdsc
	start_spark_and_run.sh 1 clustering.py /gscratch/comdata/output/reddit_similarity/subreddit_author_tf_similarities_10000.parquet /gscratch/comdata/output/reddit_clustering/subreddit_author_tf_similarities_10000.feather ---max_iter=400 --convergence_iter=15 --preference_quantile=0.5 --damping=0.85

# it's pretty difficult to get a result that isn't one huge megacluster. A sign that it's bullcrap
# /gscratch/comdata/output/reddit_clustering/wang_similarity_10000.feather:clustering.py /gscratch/comdata/output/reddit_similarity/wang_similarity_10000.feather
# 	./clustering.py /gscratch/comdata/output/reddit_similarity/wang_similarity_10000.feather /gscratch/comdata/output/reddit_clustering/wang_similarity_10000.feather ---max_iter=400 --convergence_iter=15 --preference_quantile=0.9 --damping=0.85

/gscratch/comdata/output/reddit_tsne/subreddit_author_tf_similarities_10000.feather:fit_tsne.py /gscratch/comdata/output/reddit_similarity/subreddit_author_tf_similarities_10000.parquet

	start_spark_and_run.sh 1 fit_tsne.py --similarities=/gscratch/comdata/output/reddit_similarity/subreddit_author_tf_similarities_10000.parquet --output=/gscratch/comdata/output/reddit_tsne/subreddit_author_tf_similarities_10000.feather


# /gscratch/comdata/output/reddit_tsne/wang_similarity_10000.feather:fit_tsne.py /gscratch/comdata/output/reddit_similarity/wang_similarity_10000.feather

# 	python3 fit_tsne.py --similarities=/gscratch/comdata/output/reddit_similarity/wang_similarity_10000.feather --output=/gscratch/comdata/output/reddit_tsne/wang_similarity_10000.feather

/gscratch/comdata/output/reddit_tsne/comment_authors_10000.feather:clustering.py /gscratch/comdata/output/reddit_similarity/comment_authors_10000.feather
#	$srun_cdsc python3
	start_spark_and_run.sh 1 fit_tsne.py --similarities=/gscratch/comdata/output/reddit_similarity/comment_authors_10000.feather --output=/gscratch/comdata/output/reddit_tsne/comment_authors_10000.feather
