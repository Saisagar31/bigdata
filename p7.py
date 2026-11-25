from pyspark import SparkContext

sc = SparkContext("local", "RetailApp")
sc.setLogLevel("ERROR")

products = sc.broadcast({"P101": "Laptop", "P102": "Mobile", "P103": "Headphones"})
bad = sc.accumulator(0)

data = [
    "T1,P101,2,50000",
    "T2,P999,1,500",
    "T3,P102,x,20000",
    "T4,P103,3,1500",
    "T5,P102,-1,20000",
    "broken line"
]

def check(line):
    try:
        t, pid, q, p = line.split(",")
        q, p = int(q), float(p)
        assert q > 0 and p > 0
        return (t, products.value[pid], q, p, q * p)
    except Exception:
        bad.add(1)

valid = sc.parallelize(data).map(check).filter(bool)

print("Valid Transactions:")
for row in valid.collect():
    print(row)

print("Invalid records =", bad.value)
sc.stop()
