from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("MovieRecommendation").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

data = [
    (1, 101, 5.0), (1, 102, 3.0), (1, 103, 4.0),
    (2, 101, 4.0), (2, 102, 2.0), (2, 104, 5.0),
    (3, 101, 2.0), (3, 103, 5.0), (3, 104, 4.0),
    (4, 102, 4.0), (4, 103, 3.0), (4, 104, 5.0),
    (5, 101, 5.0), (5, 102, 4.0), (5, 104, 3.0)
]

df = spark.createDataFrame(data, ["UserID", "MovieID", "Rating"])
print("Movie Ratings Dataset")
df.show()

train, test = df.randomSplit([0.8, 0.2], seed=42)

model = ALS(
    userCol="UserID", itemCol="MovieID", ratingCol="Rating",
    rank=2, maxIter=10, regParam=0.1, coldStartStrategy="drop", seed=42
).fit(train)

pred = model.transform(test)
print("Predicted Movie Ratings")
pred.show()

print("Root Mean Squared Error (RMSE):", RegressionEvaluator(labelCol="Rating").evaluate(pred))

print("Top 3 Movie Recommendations")
model.recommendForAllUsers(3).show(truncate=False)

spark.stop()
