import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
# from sklearn.cluster import KMeans
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

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
    #"bank-mini.csv", header=True,
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
# df2.show()

df2 = df2.na.drop()
# df2.show()

vecAssembler = VectorAssembler(inputCols=FEATURES_COL, outputCol="features")
df_kmeans = vecAssembler.transform(df2).select('features')
df_kmeans.show(20, False)

# Evaluation de K
# cost = np.zeros(20)
# for k in range(2,20):s
#     kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features");
#     model = kmeans.fit(df_kmeans)
#
#     predictions = model.transform(df_kmeans)
#
#     evaluator = ClusteringEvaluator()
#
#     silhouette = evaluator.evaluate(predictions)
#     cost[k] = silhouette
# # print("Silhouette with squared euclidean distance = " + str(silhouette))
#
# fig, ax = plt.subplots(1,1, figsize =(8,6))
# ax.plot(range(2,20),cost[2:20])
# ax.set_xlabel('k')
# ax.set_ylabel('cost')
# fig.show()

# Détection du centre des cluster
k = 3
kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
model = kmeans.fit(df_kmeans)
centers = model.clusterCenters()

print("Cluster Centers: ")
for center in centers:
    print(center)

transformed = model.transform(df_kmeans).select('prediction')
rows = transformed.collect()
print(rows)

df_pred = spark.createDataFrame(rows)
df_pred.show()

#df_pred = df_pred.join(df2)
#df_pred.show()

# enfaite on veut bien une colonne ID
df_pred = df_pred.withColumn("id", F.monotonically_increasing_id())
df_pred.show()

pddf_pred = df_pred.toPandas().set_index('id')
pddf_pred.head()

#threedee = plt.figure(figsize=(12,10)).gca(projection='3d')
#threedee.scatter(pddf_pred.age, pddf_pred.balance, pddf_pred.duration,  c=pddf_pred.prediction)
#threedee.set_xlabel('age')
#threedee.set_ylabel('balance')
#threedee.set_zlabel('duration')
#plt.show()

from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 
fig = plt.figure()

ax = Axes3D(fig) #<-- Note the difference from your original code...

#X, Y, Z = axes3d.get_test_data(0.05)
cset = ax.contour(df2.age, df2.balance, df2.duration,  pddf_pred.prediction, extend3d=True)
ax.clabel(cset, fontsize=9, inline=1)
ax.set_xlabel(xlabel="patate")
ax.set_ylabel(ylabel="orange")
ax.set_zlabel(zlabel="Mé nan !")
plt.show()

sc.stop()
# ----------------------------------------
#                K-means
# ----------------------------------------


# kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
# pred_y = kmeans.fit_predict(df_kmeans)
# plt.scatter(df_kmeans[:,0], df_kmeans[:,1])
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
# plt.show()
