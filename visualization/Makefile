all: subreddit_author_tf_similarities_10000.html #comment_authors_10000.html

# wang_tsne_10000.html
# wang_tsne_10000.html:/gscratch/comdata/output/reddit_tsne/wang_similarity_10000.feather /gscratch/comdata/output/reddit_clustering/wang_similarity_10000.feather tsne_vis.py
# 	python3 tsne_vis.py --tsne_data=/gscratch/comdata/output/reddit_tsne/wang_similarity_10000.feather --clusters=/gscratch/comdata/output/reddit_clustering/wang_similarity_10000.feather --output=wang_tsne_10000.html

# comment_authors_10000.html:/gscratch/comdata/output/reddit_tsne/comment_authors_10000.feather /gscratch/comdata/output/reddit_clustering/comment_authors_10000.feather tsne_vis.py
# 	python3 tsne_vis.py --tsne_data=/gscratch/comdata/output/reddit_similarity/comment_authors_10000.feather --clusters=/gscratch/comdata/output/reddit_clustering/comment_authors_10000.feather --output=comment_authors_10000.html

subreddit_author_tf_similarities_10000.html:/gscratch/comdata/output/reddit_tsne/subreddit_author_tf_similarities_10000.feather /gscratch/comdata/output/reddit_clustering/subreddit_author_tf_similarities_10000.feather tsne_vis.py
	start_spark_and_run.sh 1 tsne_vis.py --tsne_data=/gscratch/comdata/output/reddit_tsne/subreddit_author_tf_similarities_10000.feather --clusters=/gscratch/comdata/output/reddit_clustering/subreddit_author_tf_similarities_10000.feather --output=subreddit_author_tf_similarities_10000.html
