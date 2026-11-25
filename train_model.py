from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

spark = SparkSession.builder.appName("FraudTrain").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

data = [
    (1000.0, 0, 0), (2000.0, 0, 0), (3000.0, 0, 0), (50000.0, 1, 1),
    (75000.0, 1, 1), (90000.0, 1, 1), (1500.0, 1, 0), (65000.0, 0, 1)
]

df = spark.createDataFrame(data, ["Amount", "ForeignTransaction", "Fraud"])
df = VectorAssembler(inputCols=["Amount", "ForeignTransaction"], outputCol="features").transform(df)

model = LogisticRegression(labelCol="Fraud", maxIter=20).fit(df)
model.write().overwrite().save("fraud_model")

print("Fraud Detection Model Trained and Saved Successfully")
spark.stop()
