from pyspark.sql import functions as f
from pyspark.sql import Window
from pyspark.sql import SparkSession
import numpy as np

spark = SparkSession.builder.getOrCreate()
df = spark.read.text("/gscratch/comdata/users/nathante/reddit_comment_ngrams_10p_sample/")

df = df.withColumnRenamed("value","phrase")

# count phrase occurrances
phrases = df.groupby('phrase').count()
phrases = phrases.withColumnRenamed('count','phraseCount')
phrases = phrases.filter(phrases.phraseCount > 10)


# count overall
N = phrases.select(f.sum(phrases.phraseCount).alias("phraseCount")).collect()[0].phraseCount

print(f'analyzing PMI on a sample of {N} phrases') 
logN = np.log(N)
phrases = phrases.withColumn("phraseLogProb", f.log(f.col("phraseCount")) - logN)

# count term occurrances
phrases = phrases.withColumn('terms',f.split(f.col('phrase'),' '))
terms = phrases.select(['phrase','phraseCount','phraseLogProb',f.explode(phrases.terms).alias('term')])

win = Window.partitionBy('term')
terms = terms.withColumn('termCount',f.sum('phraseCount').over(win))
terms = terms.withColumnRenamed('count','termCount')
terms = terms.withColumn('termLogProb',f.log(f.col('termCount')) - logN)

terms = terms.groupBy(terms.phrase, terms.phraseLogProb, terms.phraseCount).sum('termLogProb')
terms = terms.withColumnRenamed('sum(termLogProb)','termsLogProb')
terms = terms.withColumn("phrasePWMI", f.col('phraseLogProb') - f.col('termsLogProb'))

# join phrases to term counts


df = terms.select(['phrase','phraseCount','phraseLogProb','phrasePWMI'])

df = df.sort(['phrasePWMI'],descending=True)
df = df.sortWithinPartitions(['phrasePWMI'],descending=True)
df.write.parquet("/gscratch/comdata/users/nathante/reddit_comment_ngrams_pwmi.parquet/",mode='overwrite',compression='snappy')

df = spark.read.parquet("/gscratch/comdata/users/nathante/reddit_comment_ngrams_pwmi.parquet/")

df.write.csv("/gscratch/comdata/users/nathante/reddit_comment_ngrams_pwmi.csv/",mode='overwrite',compression='none')

df = spark.read.parquet("/gscratch/comdata/users/nathante/reddit_comment_ngrams_pwmi.parquet")
df = df.select('phrase','phraseCount','phraseLogProb','phrasePWMI')

# choosing phrases occurring at least 3500 times in the 10% sample (35000 times) and then with a PWMI of at least 3 yeids about 65000 expressions.
#
df = df.filter(f.col('phraseCount') > 3500).filter(f.col("phrasePWMI")>3)
df = df.toPandas()
df.to_feather("/gscratch/comdata/users/nathante/reddit_multiword_expressions.feather")
df.to_csv("/gscratch/comdata/users/nathante/reddit_multiword_expressions.csv")
