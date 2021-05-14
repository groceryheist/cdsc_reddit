#all: /gscratch/comdata/output/reddit_similarity/tfidf/comment_terms_130k.parquet /gscratch/comdata/output/reddit_similarity/tfidf/comment_authors_130k.parquet /gscratch/comdata/output/reddit_similarity/tfidf_weekly/comment_terms_130k.parquet /gscratch/comdata/output/reddit_similarity/tfidf_weekly/comment_authors_130k.parquet
srun_singularity=source /gscratch/comdata/users/nathante/cdsc_reddit/bin/activate && srun_singularity.sh
srun_singularity_huge=source /gscratch/comdata/users/nathante/cdsc_reddit/bin/activate && srun_singularity_huge.sh
base_data=/gscratch/comdata/output
similarity_data=${base_data}/reddit_similarity
tfidf_data=${similarity_data}/tfidf
tfidf_weekly_data=${similarity_data}/tfidf_weekly
similarity_weekly_data=${similarity_data}/weekly
lsi_components=[10,50,100,200,300,400,500,600,700,850,1000,1500]

lsi_similarities: ${similarity_data}/subreddit_comment_terms_10k_LSI ${similarity_data}/subreddit_comment_authors-tf_10k_LSI ${similarity_data}/subreddit_comment_authors_10k_LSI ${similarity_data}/subreddit_comment_terms_30k_LSI ${similarity_data}/subreddit_comment_authors-tf_30k_LSI ${similarity_data}/subreddit_comment_authors_30k_LSI

all: ${tfidf_data}/comment_terms_100k.parquet ${tfidf_data}/comment_terms_30k.parquet ${tfidf_data}/comment_terms_10k.parquet ${tfidf_data}/comment_authors_100k.parquet ${tfidf_data}/comment_authors_30k.parquet ${tfidf_data}/comment_authors_10k.parquet ${similarity_data}/subreddit_comment_authors_30k.feather ${similarity_data}/subreddit_comment_authors_10k.feather  ${similarity_data}/subreddit_comment_terms_10k.feather ${similarity_data}/subreddit_comment_terms_30k.feather ${similarity_data}/subreddit_comment_authors-tf_30k.feather ${similarity_data}/subreddit_comment_authors-tf_10k.feather ${similarity_data}/subreddit_comment_terms_100k.feather ${similarity_data}/subreddit_comment_authors_100k.feather ${similarity_data}/subreddit_comment_authors-tf_100k.feather ${similarity_weekly_data}/comment_terms.parquet

#${tfidf_weekly_data}/comment_terms_100k.parquet ${tfidf_weekly_data}/comment_authors_100k.parquet ${tfidf_weekly_data}/comment_terms_30k.parquet ${tfidf_weekly_data}/comment_authors_30k.parquet ${similarity_weekly_data}/comment_terms_100k.parquet ${similarity_weekly_data}/comment_authors_100k.parquet  ${similarity_weekly_data}/comment_terms_30k.parquet ${similarity_weekly_data}/comment_authors_30k.parquet

# /gscratch/comdata/output/reddit_similarity/subreddit_comment_authors_130k.parquet /gscratch/comdata/output/reddit_similarity/subreddit_comment_authors_130k.parquet /gscratch/comdata/output/reddit_similarity/subreddit_author_tf_similarities_130k.parquet /gscratch/comdata/output/reddit_similarity/subreddit_comment_terms_130k.parquet /gscratch/comdata/output/reddit_similarity/comment_terms_weekly_130k.parquet

# all: /gscratch/comdata/output/reddit_similarity/subreddit_comment_terms_25000.parquet /gscratch/comdata/output/reddit_similarity/subreddit_comment_authors_25000.parquet /gscratch/comdata/output/reddit_similarity/subreddit_comment_authors_10000.parquet /gscratch/comdata/output/reddit_similarity/comment_terms_10000_weekly.parquet

${similarity_weekly_data}/comment_terms.parquet: weekly_cosine_similarities.py similarities_helper.py /gscratch/comdata/output/reddit_ngrams/comment_terms.parquet ${similarity_data}/subreddits_by_num_comments.csv ${tfidf_weekly_data}/comment_terms.parquet
	${srun_singularity} python3 weekly_cosine_similarities.py terms --topN=10000 --outfile=${similarity_weekly_data}/comment_terms.parquet

${similarity_data}/subreddit_comment_terms_10k.feather: ${tfidf_data}/comment_terms_100k.parquet similarities_helper.py
	${srun_singularity} python3 cosine_similarities.py term --outfile=${similarity_data}/subreddit_comment_terms_10k.feather --topN=10000

${similarity_data}/subreddit_comment_terms_10k_LSI: ${tfidf_data}/comment_terms_100k.parquet similarities_helper.py
	${srun_singularity} python3 lsi_similarities.py term --outfile=${similarity_data}/subreddit_comment_terms_10k_LSI --topN=10000 --n_components=${lsi_components} --min_df=200

${similarity_data}/subreddit_comment_terms_30k_LSI: ${tfidf_data}/comment_terms_100k.parquet similarities_helper.py
	${srun_singularity} python3 lsi_similarities.py term --outfile=${similarity_data}/subreddit_comment_terms_30k_LSI --topN=30000 --n_components=${lsi_components} --min_df=200

${similarity_data}/subreddit_comment_terms_30k.feather: ${tfidf_data}/comment_terms_30k.parquet similarities_helper.py
	${srun_singularity} python3 cosine_similarities.py term --outfile=${similarity_data}/subreddit_comment_terms_30k.feather --topN=30000

${similarity_data}/subreddit_comment_authors_30k.feather: ${tfidf_data}/comment_authors_30k.parquet similarities_helper.py
	${srun_singularity} python3 cosine_similarities.py author --outfile=${similarity_data}/subreddit_comment_authors_30k.feather --topN=30000

${similarity_data}/subreddit_comment_authors_10k.feather: ${tfidf_data}/comment_authors_10k.parquet similarities_helper.py
	${srun_singularity} python3 cosine_similarities.py author --outfile=${similarity_data}/subreddit_comment_authors_10k.feather --topN=10000

${similarity_data}/subreddit_comment_authors_10k_LSI: ${tfidf_data}/comment_authors_100k.parquet similarities_helper.py
	${srun_singularity} python3 lsi_similarities.py author --outfile=${similarity_data}/subreddit_comment_authors_10k_LSI --topN=10000 --n_components=${lsi_components} --min_df=2

${similarity_data}/subreddit_comment_authors_30k_LSI: ${tfidf_data}/comment_authors_100k.parquet similarities_helper.py
	${srun_singularity} python3 lsi_similarities.py author --outfile=${similarity_data}/subreddit_comment_authors_30k_LSI --topN=30000 --n_components=${lsi_components} --min_df=2

${similarity_data}/subreddit_comment_authors-tf_30k.feather: ${tfidf_data}/comment_authors_30k.parquet similarities_helper.py
	${srun_singularity} python3 cosine_similarities.py author-tf --outfile=${similarity_data}/subreddit_comment_authors-tf_30k.feather --topN=30000

${similarity_data}/subreddit_comment_authors-tf_10k.feather: ${tfidf_data}/comment_authors_10k.parquet similarities_helper.py
	${srun_singularity} python3 cosine_similarities.py author-tf --outfile=${similarity_data}/subreddit_comment_authors-tf_10k.feather --topN=10000

${similarity_data}/subreddit_comment_authors-tf_10k_LSI: ${tfidf_data}/comment_authors_100k.parquet similarities_helper.py
	${srun_singularity} python3 lsi_similarities.py author-tf --outfile=${similarity_data}/subreddit_comment_authors-tf_10k_LSI --topN=10000 --n_components=${lsi_components} --min_df=2

${similarity_data}/subreddit_comment_authors-tf_30k_LSI: ${tfidf_data}/comment_authors_100k.parquet similarities_helper.py
	${srun_singularity} python3 lsi_similarities.py author-tf --outfile=${similarity_data}/subreddit_comment_authors-tf_30k_LSI --topN=30000 --n_components=${lsi_components} --min_df=2

${similarity_data}/subreddit_comment_terms_100k.feather: ${tfidf_data}/comment_terms_100k.parquet similarities_helper.py
	${srun_singularity} python3 cosine_similarities.py term --outfile=${similarity_data}/subreddit_comment_terms_100k.feather --topN=100000

${similarity_data}/subreddit_comment_authors_100k.feather: ${tfidf_data}/comment_authors_100k.parquet similarities_helper.py
	${srun_singularity} python3 cosine_similarities.py author --outfile=${similarity_data}/subreddit_comment_authors_100k.feather --topN=100000

${similarity_data}/subreddit_comment_authors-tf_100k.feather: ${tfidf_data}/comment_authors_100k.parquet similarities_helper.py
	${srun_singularity} python3 cosine_similarities.py author-tf --outfile=${similarity_data}/subreddit_comment_authors-tf_100k.feather --topN=100000

${tfidf_data}/comment_terms_100k.feather/: /gscratch/comdata/output/reddit_ngrams/comment_terms.parquet ${similarity_data}/subreddits_by_num_comments.csv
	mkdir -p ${tfidf_data}/
	start_spark_and_run.sh 4 tfidf.py terms --topN=100000 --outpath=${tfidf_data}/comment_terms_100k.feather 

${tfidf_data}/comment_terms_30k.feather: /gscratch/comdata/output/reddit_ngrams/comment_terms.parquet ${similarity_data}/subreddits_by_num_comments.csv
	mkdir -p ${tfidf_data}/
	start_spark_and_run.sh 4 tfidf.py terms --topN=30000 --outpath=${tfidf_data}/comment_terms_30k.feather

${tfidf_data}/comment_terms_10k.feather: /gscratch/comdata/output/reddit_ngrams/comment_terms.parquet ${similarity_data}/subreddits_by_num_comments.csv
	mkdir -p ${tfidf_data}/
	start_spark_and_run.sh 4 tfidf.py terms --topN=10000 --outpath=${tfidf_data}/comment_terms_10k.feather

${tfidf_data}/comment_authors_100k.feather: /gscratch/comdata/output/reddit_ngrams/comment_authors.parquet ${similarity_data}/subreddits_by_num_comments.csv
	mkdir -p ${tfidf_data}/
	start_spark_and_run.sh 4 tfidf.py authors --topN=100000 --outpath=${tfidf_data}/comment_authors_100k.feather

${tfidf_data}/comment_authors_10k.parquet: /gscratch/comdata/output/reddit_ngrams/comment_authors.parquet ${similarity_data}/subreddits_by_num_comments.csv
	mkdir -p ${tfidf_data}/
	start_spark_and_run.sh 4 tfidf.py authors --topN=10000 --outpath=${tfidf_data}/comment_authors_10k.parquet

${tfidf_data}/comment_authors_30k.parquet: /gscratch/comdata/output/reddit_ngrams/comment_authors.parquet ${similarity_data}/subreddits_by_num_comments.csv
	mkdir -p ${tfidf_data}/
	start_spark_and_run.sh 4 tfidf.py authors --topN=30000 --outpath=${tfidf_data}/comment_authors_30k.parquet

${tfidf_data}/tfidf_weekly/comment_terms_100k.parquet: /gscratch/comdata/output/reddit_ngrams/comment_terms.parquet ${similarity_data}/subreddits_by_num_comments.csv
	start_spark_and_run.sh 4 tfidf.py terms_weekly --topN=100000 --outpath=${similarity_data}/tfidf_weekly/comment_authors_100k.parquet

${tfidf_data}/tfidf_weekly/comment_authors_100k.parquet: /gscratch/comdata/output/reddit_ngrams/comment_terms.parquet ${similarity_data}/subreddits_by_ppnum_comments.csv
	start_spark_and_run.sh 4 tfidf.py authors_weekly --topN=100000 --outpath=${tfidf_weekly_data}/comment_authors_100k.parquet

${tfidf_weekly_data}/comment_terms_30k.parquet:  /gscratch/comdata/output/reddit_ngrams/comment_terms.parquet ${similarity_data}/subreddits_by_num_comments.csv
	start_spark_and_run.sh 2 tfidf.py terms_weekly --topN=30000 --outpath=${tfidf_weekly_data}/comment_authors_30k.parquet

${tfidf_weekly_data}/comment_authors_30k.parquet: /gscratch/comdata/output/reddit_ngrams/comment_terms.parquet ${similarity_data}/subreddits_by_num_comments.csv
	start_spark_and_run.sh 4 tfidf.py authors_weekly --topN=30000 --outpath=${tfidf_weekly_data}/comment_authors_30k.parquet

${similarity_weekly_data}/comment_terms_100k.parquet: weekly_cosine_similarities.py similarities_helper.py ${tfidf_weekly_data}/comment_terms_100k.parquet
	${srun_singularity} python3 weekly_cosine_similarities.py terms --topN=100000 --outfile=${similarity_weekly_data}/comment_authors_100k.parquet

${similarity_weekly_data}/comment_authors_100k.parquet: weekly_cosine_similarities.py similarities_helper.py /gscratch/comdata/output/reddit_ngrams/comment_terms.parquet ${similarity_data}/subreddits_by_num_comments.csv ${tfidf_weekly_data}/comment_authors_100k.parquet
	${srun_singularity} python3 weekly_cosine_similarities.py authors --topN=100000 --outfile=${similarity_weekly_data}/comment_authors_100k.parquet

${similarity_weekly_data}/comment_terms_30k.parquet: weekly_cosine_similarities.py similarities_helper.py /gscratch/comdata/output/reddit_ngrams/comment_terms.parquet ${similarity_data}/subreddits_by_num_comments.csv ${tfidf_weekly_data}/comment_terms_30k.parquet
	${srun_singularity} python3 weekly_cosine_similarities.py terms --topN=30000 --outfile=${similarity_weekly_data}/comment_authors_30k.parquet

${similarity_weekly_data}/comment_authors_30k.parquet: weekly_cosine_similarities.py similarities_helper.py /gscratch/comdata/output/reddit_ngrams/comment_terms.parquet ${similarity_data}/subreddits_by_num_comments.csv ${tfidf_weekly_data}/comment_authors_30k.parquet
	${srun_singularity} python3 weekly_cosine_similarities.py authors --topN=30000 --outfile=${similarity_weekly_data}/comment_authors_30k.parquet

# ${tfidf_weekly_data}/comment_authors_130k.parquet: tfidf.py similarities_helper.py /gscratch/comdata/output/reddit_ngrams/comment_authors.parquet /gscratch/comdata/output/reddit_similarity/subreddits_by_num_comments.csv
# 	start_spark_and_run.sh 1 tfidf.py authors_weekly --topN=130000

# /gscratch/comdata/output/reddit_similarity/comment_authors_10000.parquet: cosine_similarities.py similarities_helper.py /gscratch/comdata/output/reddit_similarity/tfidf/comment_authors.parquet /gscratch/comdata/output/reddit_similarity/tfidf/comment_authors.parquet 
# 	start_spark_and_run.sh 1 cosine_similarities.py author --outfile=/gscratch/comdata/output/reddit_similarity/comment_authors_10000.feather

# /gscratch/comdata/output/reddit_similarity/comment_terms.parquet: cosine_similarities.py similarities_helper.py /gscratch/comdata/output/reddit_similarity/tfidf/comment_terms.parquet
# 	start_spark_and_run.sh 1 cosine_similarities.py term --outfile=/gscratch/comdata/output/reddit_similarity/comment_terms_10000.feather

# /gscratch/comdata/output/reddit_similarity/comment_terms_10000_weekly.parquet: cosine_similarities.py ${tfidf_weekly_data}/comment_authors.parquet
# 	start_spark_and_run.sh 1 weekly_cosine_similarities.py term --outfile=/gscratch/comdata/output/reddit_similarity/subreddit_comment_terms_10000_weely.parquet

# /gscratch/comdata/output/reddit_similarity/subreddit_author_tf_similarities_10000.parquet: cosine_similarities.py similarities_helper.py /gscratch/comdata/output/reddit_similarity/tfidf/comment_authors.parquet /gscratch/comdata/output/reddit_similarity/tfidf/comment_authors.parquet
# 	start_spark_and_run.sh 1 cosine_similarities.py author-tf --outfile=/gscratch/comdata/output/reddit_similarity/subreddit_author_tf_similarities_10000.parquet
