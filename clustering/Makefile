#srun_cdsc='srun -p comdata-int -A comdata --time=300:00:00 --time-min=00:15:00 --mem=100G --ntasks=1 --cpus-per-task=28'
all:/gscratch/comdata/output/reddit_clustering/comment_authors_10000.feather /gscratch/comdata/output/reddit_clustering/comment_terms_10000.feather

/gscratch/comdata/output/reddit_clustering/comment_authors_10000.feather:clustering.py /gscratch/comdata/output/reddit_similarity/comment_authors_10000.feather
#	$srun_cdsc python3
	./clustering.py /gscratch/comdata/output/reddit_similarity/comment_authors_10000.feather /gscratch/comdata/output/reddit_clustering/comment_authors_10000.feather ---max_iter=400 --convergence_iter=15 --preference_quantile=0.85 --damping=0.85

/gscratch/comdata/output/reddit_clustering/comment_terms_10000.feather:clustering.py /gscratch/comdata/output/reddit_similarity/comment_terms_10000.feather
#	$srun_cdsc python3
	./clustering.py /gscratch/comdata/output/reddit_similarity/comment_terms_10000.feather /gscratch/comdata/output/reddit_clustering/comment_terms_10000.feather ---max_iter=1000 --convergence_iter=15 --preference_quantile=0.9 --damping=0.5
