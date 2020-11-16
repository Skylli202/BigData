from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


# textFile = spark.read.csv("bank.csv")



# path = "C:/Users/anaki/Documents/Université Dijon/5A/BigData/bank.csv"
textFile = spark.read.csv(
    "bank.csv", header=True, mode="DROPMALFORMED"
)
print("spark.count: ", textFile.count())