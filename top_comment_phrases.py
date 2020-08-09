from pyspark.sql import functions as f
from pyspark.sql import Window
from pyspark.sql import SparkSession
import numpy as np

spark = SparkSession.builder.getOrCreate()
df = spark.read.text("/gscratch/comdata/users/nathante/reddit_comment_ngrams_10p_sample/")

df = df.withColumnRenamed("value","phrase")


# count overall
N = df.count()
print(f'analyzing PMI on a sample of {N} phrases') 
logN = np.log(N)

# count phrase occurrances
phrases = df.groupby('phrase').count()
phrases = phrases.withColumnRenamed('count','phraseCount')
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

df = df.repartition('phrasePWMI')
df = df.sort(['phrasePWMI'],descending=True)
df = df.sortWithinPartitions(['phrasePWMI'],descending=True)
df.write.parquet("/gscratch/comdata/users/nathante/reddit_comment_ngrams_pwmi.parquet/",mode='overwrite',compression='snappy')
df.write.csv("/gscratch/comdata/users/nathante/reddit_comment_ngrams_pwmi.csv/",mode='overwrite',compression='none')
