from pyspark.sql import SparkSession

logFile = "../data/test_data.txt"  # Should be some file on your system
spark = SparkSession.builder.appName("SimpleApp").getOrCreate()

# file = open(logFile, 'r').readlines()
# print(file)

logData = spark.read.text(logFile).cache()


numAs = logData.filter(logData.value.contains('a')).count()
numBs = logData.filter(logData.value.contains('b')).count()

print("Lines with a: %i, lines with b: %i" % (numAs, numBs))

spark.stop()