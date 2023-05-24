---
title: Utilities for Reddit Data Science
---


The reddit_cdsc project contains tools for working with Reddit data.  The project is designed for the hyak super computing system at The University of Washington.  It consists of a set of python and bash scripts and uses the [Pyspark](https://spark.apache.org/docs/latest/api/python/index.html "Pyspark documentation") and [pyarrow](https://arrow.apache.org/docs/python/ "documentation of python arrow bindings") to process large datasets.  As of November 1st 2020, the project is under active development by [Nate TeBlunthuis](https://wiki.communitydata.science/People#Nathan_TeBlunthuis_.28University_of_Washington.29 "Nate's profile on the Community Data Science Collective Wiki") and provides scripts for:

- Pulling and updating dumps from [Pushshift](https://pushshift.io "Pushshift.io") in `pull_pushshift_comments.sh` and `pull_pushshift_submissions.sh`.
- Uncompressing and parsing the dumps into [Parquet](https://parquet.apache.org/ "apahce parquet website") [datasets](https://wiki.communitydata.science/CommunityData:Hyak_Datasets#Reading_Reddit_parquet_datasets "Wikilink to documentation on the Reddit parquet datasets").
- Running text analysis based on [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf "Wikipedia article on tf-idf") including 
  - Extracting terms from Reddit comments in `tf_comments.py`
  - Detecting common phrases based on [Pointwise mutual information](https://en.wikipedia.org/wiki/Pointwise_mutual_information) "Wikipedia article on pointwise mutual information")
  - Building TF-IDF vectors for each subreddit `idf_comments.py` and (more experimentally) at the subreddit-week level `idf_comments_weekly.py` 
  - Computing cosine similarities between subreddits based on TF-IDF `term_cosine_similarity.py`. 

Right now, two steps are still in earlier stages of progress:

- Approach comparable to tf-idf for similarity between subreddits in terms of comment authors. 
- Clustering subreddits based on cosine-similarities using [power iteration clustering (PIC)](http://www.cs.cmu.edu/~wcohen/postscript/icml2010-pic-final.pdf "Paper on power iteration clustering")

The TF-IDF for comments still has some kinks to iron out to remove hyper links and bot comments. Right now subreddits that have similar automoderation messages appear very similar.

The user interfaces for most of the scripts are pretty crappy and need to be refined for re-use by others. 

## Pulling data from [Pushshift](https://pushshift.io "Pushshift.io") ##

- `pull_pushshift_comments.sh` uses wget to download comment dumps to  `/gscratch/comdata/raw_data/reddit_dumps/comments`. It doesn't download files that already exists and runs `check_comments_shas.sh` to verify the files downloaded correctly. 

- `pull_pushshift_submissions.sh` does the same for submissions and puts them in `/gscratch/comdata/raw_data/reddit_dumps/comments`.

## Building Parquet Datasets ##

Pushshift dumps are huge compressed json files with a lot of metadata that we may not need. It isn't indexed so it's expensive to pull data from just a handful of subreddits. It also turns out that it's a pain to read these compressed files straight into spark. Extracting useful variables from the dumps and building parquet datasets will make them easier to work with.  This happens in two steps:

1. Extracting json into (temporary, unpartitioned) parquet files using pyarrow.
2. Repartitioning and sorting the data using pyspark.

The final datasets live in `/gscratch/comdata/output` on Hyak..

- `reddit_comments_by_author.parquet` has comments partitioned and sorted by username (lowercase).
- `reddit_comments_by_subreddit.parquet` has comments partitioned and sorted by subreddit name (lowercase).
- `reddit_submissions_by_author.parquet` has submissions partitioned and sorted by username (lowercase).
- `reddit_submissions_by_subreddit.parquet` has submissions partitioned and sorted by subreddit name (lowercase).

Breaking this down into two steps is useful because it allows us to decompress and parse the dumps in the backfill queue and then sort them in spark. Partitioning the data makes it possible to efficiently read data for specific subreddits or authors.  Sorting it means that you can efficiently compute agreggations at the subreddit or user level. More documentation on using these files is available [here](https://wiki.communitydata.science/CommunityData:Hyak_Datasets#Reading_Reddit_parquet_datasets "Wikilink to documentation on the Reddit parquet datasets").

## TF-IDF Subreddit Similarity ##

[TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf "Wikipedia article on tf-idf") is common and simple information retrieval technique that we can use to quantify the topic of a subreddit.  The goal of TF-IDF is to build a vector for each subreddit that scores every term (or phrase) according to how characteristic it is of the overall lexicon used in that subreddit. For example, the most characteristic terms in the subreddit /r/christianity in the current version of the TF-IDF model are:

| Term         | tf_idf |
|:------------:|:------:|
| christians   | 0.581  |
| christianity | 0.569  |
| kjv          | 0.568  |
| bible        | 0.557  |
| scripture    | 0.55   |

TF-IDF stands for "term frequency - inverse document frequency" because it is the product of two terms "term frequency" and "inverse document frequency." Term frequency quantifies the amount that a term appears in a subreddit (document). Inverse document frequency quantifies how much that term appears in other subreddits (documents). As you can see on the Wikipedia page, there are many possible ways of constructing and combining these terms. 

$x + y = z_{1,d}$ 

I chose to normalize term frequency by the maximum (raw) term frequency for each subreddit:
$\mathrm{tf}_{t,d} = \frac{f_{t,d}}{\sum_{t^{'} \in d}{f_{t^{'},d}}}$ 

I use the log inverse document frequency:
$\mathrm{idf}_{t} = log\frac{N}{| {d \in D : t \in d} |}$

I then combine them using some smoothing to get:

$\mathrm{tfidf}_{t,d} = (0.5 + 0.5 \cdot \mathrm{tf}_{t,d}) \cdot \mathrm{idf}_{t}$ 

### Building TF-IDF vectors ###

The process for building TF-IDF vectors has four steps:

1. Extracting terms using `tf_comments.py`
2. Detecting common phrases using `top_comment_phrases.py`
3. Extracting terms and common phrases using `tf_comments.py --mwe-pass='second'`
4. Building idf and tf-idf scores in `idf_comments.py`

#### Running `tf_comments.py` on the backfill queue ####

The main reason that I did it in 4 steps instead of one is to take advantage of the backfill queue for running `tf_comments.py`.  This step requires reading all of the text in every comment and converting it to a bag of words at the subreddit-level.  This is a lot of computation that is easily parallelizable. The script `run_tf_jobs.sh` partially automates running steps 1 (or 3) on the backfill queue. 

#### Phrase detection using Pointwise Mutual Information ####

TF-IDF is simple, but only uses single words (unigrams).  Sequences of multiple words can be important to account for how words have different meanings in different contexts or how sequences of words refer to distinct things like names. Dealing with context or longer sequences of words is a common challenge in natural language processing since the number of possible n-grams grows like crazy as n gets bigger. Phrase detection helps this  problem by limiting the set of n-grams to those most informative. 

But how do we detect phrases?  I implemented [Pointwise mutual information](https://en.wikipedia.org/wiki/Pointwise_mutual_information) "Wikipedia article on pointwise mutual information"), which is a pretty simple way, but seems to work pretty well. 

PMI is an quantity derived from information theory. The intuition is that if two words occur together quite frequently compared to how often they appear separately then the cooccurrance is likely to be informative. 

$\operatorname{pmi}(x;y) \equiv \log\frac{p(x,y)}{p(x)p(y)} = \log\frac{p(x|y)}{p(x)} = \log\frac{p(y|x)}{p(y)}.$

In `tf_comments.py` if `--mwe-pass=first` then a 10\% sample of 1-4-grams (sequences of terms up to length 4) will be written to a file to be consumed by `top_comment_phrases.py`.  `top_comment_phrases.py` computes the PMI for these possible phrases and writes those that occur at least 3500 times in the sample of n-grams and have a PWMI of at least 3 (about 65000 expressions). 

`tf_comments.py --mwe-pass=second` then uses the detected phrases and adds them to the term frequency data. 

### Cosine Similarity ###

Once the tf-idf vectors are built, making a similarity score between two subreddits is straightforward using cosine similarity. 

$\text{similarity} = \cos(\theta) = {\mathbf{A} \cdot \mathbf{B} \over \|\mathbf{A}\| \|\mathbf{B}\|} = \frac{ \sum\limits_{i=1}^{n}{A_i  B_i} }{ \sqrt{\sum\limits_{i=1}^{n}{A_i^2}}  \sqrt{\sum\limits_{i=1}^{n}{B_i^2}} }$

Intuitively, we represent two subreddits as lines in a high-dimensional space (tf-idf vectors). 
In linear algebra, the dot product ($\cdot$) between two vectors takes their weighted sum (e.g. linear regression is a dot product of a vector of covariates and a vector of weights).  
The vectors might have different lengths like if one subreddit has words in comments than the other, so in cosine similarity the dot product is normalized by the magnitude (lengths) of the vectors. 
It turns out that this is equivalent to taking the cosine of the two vectors.  So cosine similarity in essence quantifies the angle between the two lines in high-dimensional space.  If the cosine similarity between two subreddits is greater then their tf-idf vectors are more correlated. 

Cosine similarity with tf-idf is popular (indeed it has been applied to Reddit in research several times before) because it quantifies the correlation between the most characteristic terms for two communities.

Compared to other approach to similarity like those using word embeddings or topic models it may struggle to handle polysemy, synonymy, or correlations between different terms.  Using phrase detection helps with this a little bit.  The advantages of this approach are simplicity and scalability.  

Therefore, we support [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis "Wikipedia article on Latent semantic analysis") as an intermediate step to improve upon similarities based on raw tf-idfs. 
LSI reduces the dimensionality of the data before computing cosine similariies. Doing so improves the face-validity of the resulting similarity measures and the results of downstream clustering analysis.
