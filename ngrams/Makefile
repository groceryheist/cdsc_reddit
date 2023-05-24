outputdir=../../data/reddit_ngrams/
inputdir=../../data/reddit_comments_by_subreddit.parquet
authors_tfdir=${outputdir}/comment_authors.parquet
srun=sbatch --wait --verbose run_job.sbatch

all: ${outputdir}/comment_authors_sorted.parquet/_SUCCESS 

tf_task_list_1: tf_comments.py
	${srun} bash -c "python3 tf_comments.py gen_task_list --mwe_pass='first' --outputdir=${outputdir} --tf_task_list=$@ --inputdir=${inputdir}"

${outputdir}/comment_terms.parquet:tf_task_list_1
	mkdir -p sbatch_log
	sbatch --wait --verbose --array=1-$(shell cat $< | wc -l) run_array.sbatch 0 $<

${outputdir}/comment_authors.parquet:${outputdir}/comment_terms.parquet
	-

${outputdir}/comment_authors_sorted.parquet:${outputdir}/comment_authors.parquet sort_tf_comments.py
	../start_spark_and_run.sh 3 sort_tf_comments.py --inparquet=$< --outparquet=$@ --colname=author

${outputdir}/comment_authors_sorted.parquet/_SUCCESS:${outputdir}/comment_authors_sorted.parquet


${inputdir}:
	$(MAKE) -C ../datasets
