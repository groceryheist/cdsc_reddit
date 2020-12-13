srun_cdsc='srun -p comdata-int -A comdata --time=300:00:00 --time-min=00:15:00 --mem=100G --ntasks=1 --cpus-per-task=28'
affinity/subreddit_comment_authors_10000.feather:clustering.py /gscratch/comdata/output/reddit_similarity/subreddit_comment_authors_10000.parquet
#	$srun_cdsc python3
	clustering.py /gscratch/comdata/output/reddit_similarity/subreddit_comment_authors_10000.feather affinity/subreddit_comment_authors_10000.feather ---max_iter=400 --convergence_iter=15 --preference_quantile=0.85 --damping=0.85
