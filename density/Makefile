all: /gscratch/comdata/output/reddit_density/comment_terms_10000.feather /gscratch/comdata/output/reddit_density/comment_authors_10000.feather

/gscratch/comdata/output/reddit_density/comment_terms_10000.feather:overlap_density.py /gscratch/comdata/output/reddit_similarity/comment_terms_10000.feather /gscratch/comdata/output/reddit_similarity/comment_terms_10000.feather
	python3 overlap_density.py terms --inpath="/gscratch/comdata/output/reddit_similarity/comment_terms_10000.feather" --outpath="/gscratch/comdata/output/reddit_density/comment_terms_10000.feather" --agg=pd.DataFrame.sum

/gscratch/comdata/output/reddit_density/comment_authors_10000.feather:overlap_density.py /gscratch/comdata/output/reddit_similarity/comment_authors_10000.feather
	python3 overlap_density.py authors --inpath="/gscratch/comdata/output/reddit_similarity/comment_authors_10000.feather" --outpath="/gscratch/comdata/output/reddit_density/comment_authors_10000.feather" --agg=pd.DataFrame.sum
