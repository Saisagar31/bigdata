from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, lower

spark = SparkSession.builder.appName("SportsAnalytics").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

lines = spark.readStream.format("socket").option("host", "localhost").option("port", 9999).load()

words = lines.select(explode(split(lower("value"), r"\W+")).alias("word"))

keys = ["kohli", "rohit", "dhoni", "sachin", "four", "six", "wicket",
        "run", "runs", "boundary", "out", "catch", "century"]

counts = words.filter(words.word.isin(keys)).groupBy("word").count()

counts.writeStream.outputMode("complete").format("console").start().awaitTermination()
