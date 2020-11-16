import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
# from sklearn.cluster import KMeans
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

def strToBool(str):
    return str.lower() in ("yes")

def init_spark():
    spark = SparkSession \
        .builder \
        .appName("BigData Project") \
        .getOrCreate()
    sc = spark.sparkContext
    return spark, sc

spark, sc = init_spark()
df = spark.read.option("delimiter", ";").csv(
    "bank.csv", header=True,
)
df.printSchema()

# Change column type
df2 = df.withColumn('age', df['age'].cast("int"))
df2 = df2.withColumn('balance', df['balance'].cast("int"))
df2 = df2.withColumn('duration', df['duration'].cast("int"))
df2 = df2.withColumn('campaign', df['campaign'].cast("int"))
df2 = df2.withColumn('pdays', df['pdays'].cast("int"))
df2 = df2.withColumn('previous', df['previous'].cast("int"))

# Remplace
df2 = df2.withColumn('default',F.when(F.col('default')=="yes", 1).otherwise(0))
df2 = df2.withColumn('housing',F.when(F.col('housing')=="yes", 1).otherwise(0))
df2 = df2.withColumn('loan',F.when(F.col('loan')=="yes", 1).otherwise(0))
df2 = df2.withColumn('y',F.when(F.col('y')=="yes", 1).otherwise(0))
df2 = df2.withColumn('contact',F.when(F.col('contact')=="cellular", 0).when(F.col('contact')=="unknown", 0.5).otherwise(1))

# Drop column that we don't care
df2 = df2.drop(df2.job)
df2 = df2.drop(df2.marital)
df2 = df2.drop(df2.education)
df2 = df2.drop(df2.day)
df2 = df2.drop(df2.month)
df2 = df2.drop(df2.poutcome)

df2.printSchema()
FEATURES_COL = ['age', 'default', 'balance', 'housing', 'loan', 'contact', 'duration', 'campaign', 'pdays', 'previous', 'y']

for col in df2.columns:
    if col in FEATURES_COL:
        df2 = df2.withColumn(col,df2[col].cast('float'))
df2.show()

df2 = df2.na.drop()
df2.show()

vecAssembler = VectorAssembler(inputCols=FEATURES_COL, outputCol="features")
df_kmeans = vecAssembler.transform(df2).select('features')
df_kmeans.show(20, False)

cost = np.zeros(20)
for k in range(2,20):
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
    model = kmeans.fit(df_kmeans.sample(False,0.1, seed=42))
    cost[k] = model.computeCost(df_kmeans)

fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,20),cost[2:20])
ax.set_xlabel('k')
ax.set_ylabel('cost')

# ----------------------------------------
#                K-means
# ----------------------------------------


# kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
# pred_y = kmeans.fit_predict(X)
# plt.scatter(X[:,0], X[:,1])
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
# plt.show()
