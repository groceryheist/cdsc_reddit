from pyspark.sql import functions as f
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.types import FloatType
import zlib

def zlib_entropy_rate(s):
    sb = s.encode()
    if len(sb) == 0:
        return None
    else:
        return len(zlib.compress(s.encode(),level=6))/len(s.encode())
    
zlib_entropy_rate_udf = f.udf(zlib_entropy_rate,FloatType())

spark = SparkSession.builder.getOrCreate()

df = spark.read.parquet("/gscratch/comdata/output/reddit_comments_by_author.parquet",compression='snappy')

df = df.withColumn("saidbot",f.lower(f.col("body")).like("%bot%"))

# df = df.filter(df.subreddit=='seattle')
# df = df.cache()
botreplies = df.filter(f.lower(df.body).rlike(".*[good|bad] bot.*"))
botreplies = botreplies.select([f.col("parent_id").substr(4,100).alias("bot_comment_id"),f.lower(f.col("body")).alias("good_bad_bot"),f.col("link_id").alias("gbbb_link_id")])
botreplies = botreplies.groupby(['bot_comment_id']).agg(f.count('good_bad_bot').alias("N_goodbad_votes"),
                                                        f.sum((f.lower(f.col('good_bad_bot')).like('%good bot%').astype("double"))).alias("n_good_votes"),
                                                        f.sum((f.lower(f.col('good_bad_bot')).like('%bad bot%').astype("double"))).alias("n_bad_votes"))

comments_by_author = df.select(['author','id','saidbot']).groupBy('author').agg(f.count('id').alias("N_comments"),
                                                                                f.mean(f.col('saidbot').astype("double")).alias("prop_saidbot"),
                                                                                f.sum(f.col('saidbot').astype("double")).alias("n_saidbot"))

# pd_comments_by_author = comments_by_author.toPandas()
# pd_comments_by_author['frac'] = 500 / pd_comments_by_author['N_comments']
# pd_comments_by_author.loc[pd_comments_by_author.frac > 1, 'frac'] = 1
# fractions = pd_comments_by_author.loc[:,['author','frac']]
# fractions = fractions.set_index('author').to_dict()['frac']

# sampled_author_comments = df.sampleBy("author",fractions).groupBy('author').agg(f.concat_ws(" ", f.collect_list('body')).alias('comments'))
df = df.withColumn("randn",f.randn(seed=1968))

win = Window.partitionBy("author").orderBy("randn")

df = df.withColumn("randRank",f.rank().over(win))
sampled_author_comments = df.filter(f.col("randRank") <= 1000)
sampled_author_comments = sampled_author_comments.groupBy('author').agg(f.concat_ws(" ", f.collect_list('body')).alias('comments'))

author_entropy_rates = sampled_author_comments.select(['author',zlib_entropy_rate_udf(f.col('comments')).alias("entropy_rate")])

parents = df.join(botreplies, on=df.id==botreplies.bot_comment_id,how='right_outer')

win1 = Window.partitionBy("author")
parents = parents.withColumn("first_bot_reply",f.min(f.col("CreatedAt")).over(win1))

first_bot_reply = parents.filter(f.col("first_bot_reply")==f.col("CreatedAt"))
first_bot_reply = first_bot_reply.withColumnRenamed("CreatedAt","FB_CreatedAt")
first_bot_reply = first_bot_reply.withColumnRenamed("id","FB_id")

comments_since_first_bot_reply = df.join(first_bot_reply,on = 'author',how='right_outer').filter(f.col("CreatedAt")>=f.col("first_bot_reply"))
comments_since_first_bot_reply = comments_since_first_bot_reply.groupBy("author").agg(f.count("id").alias("N_comments_since_firstbot"))

bots = parents.groupby(['author']).agg(f.sum('N_goodbad_votes').alias("N_goodbad_votes"),
                                          f.sum(f.col('n_good_votes')).alias("n_good_votes"),
                                          f.sum(f.col('n_bad_votes')).alias("n_bad_votes"),
                                          f.count(f.col('author')).alias("N_bot_posts"))

bots = bots.join(comments_by_author,on="author",how='left_outer')
bots = bots.join(comments_since_first_bot_reply,on="author",how='left_outer')
bots = bots.join(author_entropy_rates,on='author',how='left_outer')

bots = bots.orderBy("N_goodbad_votes",ascending=False)
bots = bots.repartition(1)
bots.write.parquet("/gscratch/comdata/output/reddit_good_bad_bot.parquet",mode='overwrite')
