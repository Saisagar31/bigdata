from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("EmployeeCRUD").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

data = [
    (101, "Rahul", "IT", 50000),
    (102, "Priya", "HR", 45000),
    (103, "Amit", "Finance", 60000),
    (104, "Sneha", "IT", 55000)
]

spark.createDataFrame(data, ["ID", "Name", "Department", "Salary"]).createOrReplaceTempView("Employee")

def run(title, query):
    print(title)
    df = spark.sql(query)
    df.show()
    df.createOrReplaceTempView("Employee")

run("Original Employee Records", "SELECT * FROM Employee")
run("After Inserting Employee", "SELECT * FROM Employee UNION ALL SELECT 105, 'Kiran', 'HR', 48000")
run("After Updating Salary", "SELECT ID, Name, Department, CASE WHEN ID = 101 THEN 65000 ELSE Salary END AS Salary FROM Employee")
run("After Deleting Employee", "SELECT * FROM Employee WHERE ID != 103")

spark.stop()
