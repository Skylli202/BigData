import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

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
    # "bank.csv", header=True,
    "bank-mini.csv", header=True,
)
print("Schema after reading CSV file :")
df.printSchema()

# Change column type
df2 = df.withColumn('age', df['age'].cast("int"))
df2 = df2.withColumn('balance', df['balance'].cast("int"))
df2 = df2.withColumn('duration', df['duration'].cast("int"))
df2 = df2.withColumn('campaign', df['campaign'].cast("int"))
df2 = df2.withColumn('pdays', df['pdays'].cast("int"))
df2 = df2.withColumn('previous', df['previous'].cast("int"))

# Replace text by integer value
df2 = df2.withColumn('default', F.when(F.col('default') == "yes", 1).otherwise(0))
df2 = df2.withColumn('housing', F.when(F.col('housing') == "yes", 1).otherwise(0))
df2 = df2.withColumn('loan', F.when(F.col('loan') == "yes", 1).otherwise(0))
df2 = df2.withColumn('y', F.when(F.col('y') == "yes", 1).otherwise(0))
df2 = df2.withColumn('contact',
                     F.when(F.col('contact') == "cellular", 0).when(F.col('contact') == "unknown", 0.5).otherwise(1))

# Drop column that we don't care
df2 = df2.drop(df2.job)
df2 = df2.drop(df2.marital)
df2 = df2.drop(df2.education)
df2 = df2.drop(df2.day)
df2 = df2.drop(df2.month)
df2 = df2.drop(df2.poutcome)

FEATURES_COL = ['age',
                'default',
                'balance',
                'housing',
                'loan',
                'contact',
                'duration',
                'campaign',
                'pdays',
                'previous',
                'y']

for col in df2.columns:
    if col in FEATURES_COL:
        df2 = df2.withColumn(col, df2[col].cast('float'))

df2 = df2.na.drop()

print("Schema after pre-process for KMeans:")
df2.printSchema()

vecAssembler = VectorAssembler(inputCols=FEATURES_COL, outputCol="features")
df_kmeans = vecAssembler.transform(df2).select('features')

print("KMeans dataframe :")
df_kmeans.show(20, False)

# # Evaluation de K (to figure out which value of K we need to use)
# cost = np.zeros(20)
# for k in range(2,20):
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

# KMeans evaluation of Cluster's center location
k = 3
kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
model = kmeans.fit(df_kmeans)
centers = model.clusterCenters()

# # Print center position
# print("Cluster Centers: ")
# for center in centers:
#     print(center)

# Now, add the prediction column to our df2
transformed = model.transform(df_kmeans).select('prediction')
rows = transformed.collect()

df_pred = spark.createDataFrame(rows)
df_pred = df_pred.join(df2)

# We need an ID column, so here it is...
df_pred = df_pred.withColumn("id", F.monotonically_increasing_id())

print("Dataframe with predict, features and ID")
df_pred.show()

pddf_pred = df_pred.toPandas().set_index('id')

print("pandas df:")
print(pddf_pred.head())

# PCA
print("\n\n---------------- PCA ----------------")
from sklearn.decomposition import PCA

pddf_numpy = pddf_pred.to_numpy()
print(pddf_numpy.shape, pddf_numpy)

pca = PCA(n_components=2)
pca.fit(pddf_numpy)
pddf_numpy_pca = pca.transform(pddf_numpy)
pddf_numpy_pca_inverse = pca.inverse_transform(pddf_numpy_pca)
print("original shape:   ", pddf_numpy.shape)
print("transformed shape:", pddf_numpy_pca.shape)
print("inverse transformed shape:", pddf_numpy_pca_inverse.shape)

# plt it !
# threedee = plt.figure(figsize=(12,10)).gca(projection='3d')

plt.scatter(pddf_numpy_pca[:, 0], pddf_numpy_pca[:, 1])
# threedee.set_xlabel('age')
# threedee.set_ylabel('balance')
# threedee.set_zlabel('duration')
plt.show()

print("\n\n\n\n\nsc.stop() :")
sc.stop()
