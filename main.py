import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
    # "bank-full.csv", header=True,
    "bank.csv", header=True,
    # "bank-mini.csv", header=True,
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

# Evaluation de K (to figure out which value of K we need to use)
cost = np.zeros(20)
for k in range(2,20):
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features");
    model = kmeans.fit(df_kmeans)

    predictions = model.transform(df_kmeans)

    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)
    cost[k] = silhouette
# print("Silhouette with squared euclidean distance = " + str(silhouette))

fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,20),cost[2:20])
ax.set_xlabel('k')
ax.set_ylabel('cost')
fig.show()

# KMeans evaluation of Cluster's center location
k = 2
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

print("ROWS :")
print(rows)

# PCA
# https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
print("\n\n---------------- PCA ----------------")
pddf_numpy = df2.toPandas().to_numpy()
print(pddf_numpy.shape, pddf_numpy)

scaler = StandardScaler()
pddf_numpy_normazized = scaler.fit_transform(pddf_numpy)
pca = PCA(n_components=2)
pddf_numpy_normazized_pca = pca.fit_transform(pddf_numpy_normazized)

colors_y = [e[0] for e in df2.select('y').toPandas().to_numpy()]
colors_predict = [e[0] for e in spark.createDataFrame(rows).select('prediction').toPandas().to_numpy()]

# plot it !
plt.scatter(pddf_numpy_normazized_pca[:, 0], pddf_numpy_normazized_pca[:, 1], c=colors_y)
plt.show()
plt.scatter(pddf_numpy_normazized_pca[:, 0], pddf_numpy_normazized_pca[:, 1], c=colors_predict)
plt.show()

print("\n\n\n\n\nsc.stop() :")
# sc.stop()
