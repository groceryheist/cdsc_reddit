#!/usr/bin/env python3
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from itertools import groupby, islice, chain
import fire
from collections import Counter
import os
import re
from nltk import wordpunct_tokenize, MWETokenizer, sent_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
import string
from random import random

# remove urls
# taken from https://stackoverflow.com/questions/3809401/what-is-a-good-regular-expression-to-match-a-url
urlregex = re.compile(r"[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)")

# compute term frequencies for comments in each subreddit by week
def weekly_tf(partition, mwe_pass = 'first'):
    dataset = ds.dataset(f'/gscratch/comdata/output/reddit_comments_by_subreddit.parquet/{partition}', format='parquet')
    if not os.path.exists("/gscratch/comdata/users/nathante/reddit_comment_ngrams_10p_sample/"):
        os.mkdir("/gscratch/comdata/users/nathante/reddit_comment_ngrams_10p_sample/")

    if not os.path.exists("/gscratch/comdata/users/nathante/reddit_tfidf_test_authors.parquet_temp/"):
        os.mkdir("/gscratch/comdata/users/nathante/reddit_tfidf_test_authors.parquet_temp/")

    ngram_output = partition.replace("parquet","txt")

    if mwe_pass == 'first':
        if os.path.exists(f"/gscratch/comdata/output/reddit_ngrams/comment_ngrams_10p_sample/{ngram_output}"):
            os.remove(f"/gscratch/comdata/output/reddit_ngrams/comment_ngrams_10p_sample/{ngram_output}")
    
    batches = dataset.to_batches(columns=['CreatedAt','subreddit','body','author'])


    schema = pa.schema([pa.field('subreddit', pa.string(), nullable=False),
                        pa.field('term', pa.string(), nullable=False),
                        pa.field('week', pa.date32(), nullable=False),
                        pa.field('tf', pa.int64(), nullable=False)]
    )

    author_schema = pa.schema([pa.field('subreddit', pa.string(), nullable=False),
                               pa.field('author', pa.string(), nullable=False),
                               pa.field('week', pa.date32(), nullable=False),
                               pa.field('tf', pa.int64(), nullable=False)]
    )

    dfs = (b.to_pandas() for b in batches)

    def add_week(df):
        df['week'] = (df.CreatedAt - pd.to_timedelta(df.CreatedAt.dt.dayofweek, unit='d')).dt.date
        return(df)

    dfs = (add_week(df) for df in dfs)

    def iterate_rows(dfs):
        for df in dfs:
            for row in df.itertuples():
                yield row

    rows = iterate_rows(dfs)

    subreddit_weeks = groupby(rows, lambda r: (r.subreddit, r.week))

    if mwe_pass != 'first':
        mwe_dataset = pd.read_feather(f'/gscratch/comdata/output/reddit_ngrams/multiword_expressions.feather')
        mwe_dataset = mwe_dataset.sort_values(['phrasePWMI'],ascending=False)
        mwe_phrases = list(mwe_dataset.phrase)
        mwe_phrases = [tuple(s.split(' ')) for s in mwe_phrases]
        mwe_tokenizer = MWETokenizer(mwe_phrases)
        mwe_tokenize = mwe_tokenizer.tokenize
    
    else:
        mwe_tokenize = MWETokenizer().tokenize

    def remove_punct(sentence):
        new_sentence = []
        for token in sentence:
            new_token = ''
            for c in token:
                if c not in string.punctuation:
                    new_token += c
            if len(new_token) > 0:
                new_sentence.append(new_token)
        return new_sentence

    stopWords = set(stopwords.words('english'))

    # we follow the approach described in datta, phelan, adar 2017
    def my_tokenizer(text):
        # remove stopwords, punctuation, urls, lower case
        # lowercase        
        text = text.lower()

        # remove urls
        text = urlregex.sub("", text)

        # sentence tokenize
        sentences = sent_tokenize(text)

        # wordpunct_tokenize
        sentences = map(wordpunct_tokenize, sentences)

        # remove punctuation
                        
        sentences = map(remove_punct, sentences)

        # remove sentences with less than 2 words
        sentences = filter(lambda sentence: len(sentence) > 2, sentences)

        # datta et al. select relatively common phrases from the reddit corpus, but they don't really explain how. We'll try that in a second phase.
        # they say that the extract 1-4 grams from 10% of the sentences and then find phrases that appear often relative to the original terms
        # here we take a 10 percent sample of sentences 
        if mwe_pass == 'first':
            sentences = list(sentences)
            for sentence in sentences:
                if random() <= 0.1:
                    grams = list(chain(*map(lambda i : ngrams(sentence,i),range(4))))
                    with open(f'/gscratch/comdata/output/reddit_ngrams/comment_ngrams_10p_sample/{ngram_output}','a') as gram_file:
                        for ng in grams:
                            gram_file.write(' '.join(ng) + '\n')
                for token in sentence:
                    if token not in stopWords:
                        yield token

        else:
            # remove stopWords
            sentences = map(mwe_tokenize, sentences)
            sentences = map(lambda s: filter(lambda token: token not in stopWords, s), sentences)
            for sentence in sentences:
                for token in sentence:
                    yield token

    def tf_comments(subreddit_weeks):
        for key, posts in subreddit_weeks:
            subreddit, week = key
            tfs = Counter([])
            authors = Counter([])
            for post in posts:
                tokens = my_tokenizer(post.body)
                tfs.update(tokens)
                authors.update([post.author])

            for term, tf in tfs.items():
                yield [True, subreddit, term, week, tf]

            for author, tf in authors.items():
                yield [False, subreddit, author, week, tf]

    outrows = tf_comments(subreddit_weeks)

    outchunksize = 10000

    with pq.ParquetWriter(f"/gscratch/comdata/output/reddit_ngrams/comment_terms.parquet/{partition}",schema=schema,compression='snappy',flavor='spark') as writer, pq.ParquetWriter(f"/gscratch/comdata/output/reddit_ngrams/comment_authors.parquet/{partition}",schema=author_schema,compression='snappy',flavor='spark') as author_writer:
    
        while True:

            chunk = islice(outrows,outchunksize)
            chunk = (c for c in chunk if c[1] is not None)
            pddf = pd.DataFrame(chunk, columns=["is_token"] + schema.names)
            author_pddf = pddf.loc[pddf.is_token == False, schema.names]
            pddf = pddf.loc[pddf.is_token == True, schema.names]
            author_pddf = author_pddf.rename({'term':'author'}, axis='columns')
            author_pddf = author_pddf.loc[:,author_schema.names]
            table = pa.Table.from_pandas(pddf,schema=schema)
            author_table = pa.Table.from_pandas(author_pddf,schema=author_schema)
            do_break = True

            if table.shape[0] != 0:
                writer.write_table(table)
                do_break = False
            if author_table.shape[0] != 0:
                author_writer.write_table(author_table)
                do_break = False

            if do_break:
                break

        writer.close()
        author_writer.close()


def gen_task_list(mwe_pass='first'):
    files = os.listdir("/gscratch/comdata/output/reddit_comments_by_subreddit.parquet/")
    with open("tf_task_list",'w') as outfile:
        for f in files:
            if f.endswith(".parquet"):
                outfile.write(f"./tf_comments.py weekly_tf --mwe-pass {mwe_pass} {f}\n")

if __name__ == "__main__":
    fire.Fire({"gen_task_list":gen_task_list,
               "weekly_tf":weekly_tf})
