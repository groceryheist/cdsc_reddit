all: ../../data/reddit_comments_by_subreddit.parquet ../../data/reddit_submissions_by_subreddit.parquet

../../data/reddit_comments_by_subreddit.parquet:../../data/temp/reddit_comments.parquet
	../start_spark_and_run.sh 4 comments_2_parquet_part2.py

../../data/temp/reddit_comments.parquet: comments_task_list.sh run_comments_jobs.sbatch
	mkdir -p comments_jobs
	mkdir -p ../../data/temp/
	sbatch --wait --array=1-$(shell cat comments_task_list.sh | wc -l) run_comments_jobs.sbatch 0

temp_reddit_comments.parquet: ../../data/temp/reddit_comments.parquet

comments_task_list.sh: comments_2_parquet_part1.py
	srun -p compute-bigmem -A comdata --nodes=1 --mem-per-cpu=9g -c 40 --time=120:00:00 bash -c "source ~/.bashrc && python3 comments_2_parquet_part1.py gen_task_list --overwrite=False"

submissions_task_list.sh: submissions_2_parquet_part1.py
	srun -p compute-bigmem -A comdata --nodes=1 --mem-per-cpu=9g -c 40 --time=120:00:00 python3 submissions_2_parquet_part1.py gen_task_list

../../data/reddit_submissions_by_subreddit.parquet:../../data/temp/reddit_submissions.parquet
	../start_spark_and_run.sh 4 submissions_2_parquet_part2.py

../../data/temp/reddit_submissions.parquet: submissions_task_list.sh run_submissions_jobs.sbatch
	mkdir -p submissions_jobs
	rm -rf ../../data/temp/reddit_submissions.parquet
	mkdir -p ../../data/temp/
	sbatch --wait --array=1-$(shell cat submissions_task_list.sh | wc -l) run_submissions_jobs.sbatch 0

temp_reddit_submissions.parquet: ../../data/temp/reddit_submissions.parquet
