srun=srun -p compute-bigmem -A comdata --mem-per-cpu=9g --time=200:00:00 -c 40
srun_huge=srun -p compute-hugemem -A comdata --mem=724g --time=200:00:00 -c 40

similarity_data=../../data/reddit_similarity
tfidf_data=${similarity_data}/tfidf
lsi_components=[10,50,100,200,300,400,500,600,700,850]

lsi_similarities: ${similarity_data}/subreddit_comment_authors-tf_10k_LSI

all: ${similarity_data}/subreddit_comment_authors-tf_10k.feather

${similarity_data}/subreddit_comment_authors-tf_10k_LSI: ${tfidf_data}/comment_authors_100k.parquet similarities_helper.py ${similarity_data}/subreddits_by_num_comments_nonsfw.csv
	 ${srun_huge} /bin/bash -c "source ~/.bashrc; python3 lsi_similarities.py author-tf --outfile=${similarity_data}/subreddit_comment_authors-tf_10k_LSI --topN=10000 --n_components=${lsi_components} --min_df=10 --inpath=$<"

${similarity_data}/subreddits_by_num_comments_nonsfw.csv: ../../data/reddit_submissions_by_subreddit.parquet ../../data/reddit_comments_by_subreddit.parquet
	../start_spark_and_run.sh 3 top_subreddits_by_comments.py

${tfidf_data}/comment_authors_100k.parquet: ../../data/reddit_ngrams/comment_authors_sorted.parquet ${similarity_data}/subreddits_by_num_comments_nonsfw.csv
	../start_spark_and_run.sh 3 tfidf.py authors --topN=100000 --inpath=$< --outpath=${tfidf_data}/comment_authors_100k.parquet

../../data/reddit_ngrams/comment_authors_sorted.parquet:
	$(MAKE) -C ../ngrams

../../data/reddit_submissions_by_subreddit.parquet:
	$(MAKE) -C ../datasets

../../data/reddit_comments_by_subreddit.parquet:
	$(MAKE) -C ../datasets
