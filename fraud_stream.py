from pyspark.sql import SparkSession
from pyspark.sql.functions import split
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegressionModel

spark = SparkSession.builder.appName("FraudDetection").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

model = LogisticRegressionModel.load("fraud_model")

lines = spark.readStream.format("socket").option("host", "localhost").option("port", 9999).load()

f = split(lines.value, ",")
data = lines.select(
    f[0].alias("TransactionID"),
    f[1].cast("double").alias("Amount"),
    f[2].cast("double").alias("ForeignTransaction")
).dropna()

data = VectorAssembler(inputCols=["Amount", "ForeignTransaction"], outputCol="features").transform(data)

result = model.transform(data).select("TransactionID", "Amount", "ForeignTransaction", "prediction")

result.writeStream.format("console").start().awaitTermination()
