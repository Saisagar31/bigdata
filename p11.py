from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, Tokenizer, HashingTF, VectorAssembler

spark = SparkSession.builder.appName("CustomerAnalytics").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.csv("customers.csv", header=True, inferSchema=True)
df.show()
df.printSchema()
print("Number of Records:", df.count())

df = df.dropna()

pipe = Pipeline(stages=[
    StringIndexer(inputCols=["Gender", "City"], outputCols=["GenderIndex", "CityIndex"]),
    Tokenizer(inputCol="Review", outputCol="Words"),
    HashingTF(inputCol="Words", outputCol="TextFeatures", numFeatures=10),
    VectorAssembler(inputCols=["GenderIndex", "CityIndex", "TextFeatures"], outputCol="Features")
])

df = pipe.fit(df).transform(df)
df.select("CustomerID", "GenderIndex", "CityIndex", "Words", "TextFeatures", "Features").show(truncate=False)

spark.stop()
