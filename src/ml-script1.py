from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation
from pyspark.sql import SparkSession

import os

# os.environ['PYSPARK_DRIVER_PYTHON'] = '/Library/Frameworks/Python.framework/Versions/3.7/bin/python3'
os.environ['PYSPARK_PYTHON'] = '/Library/Frameworks/Python.framework/Versions/3.6/bin/python3'


spark = SparkSession \
        .builder \
        .appName("ml-script1") \
        .getOrCreate()


data = [(Vectors.sparse(5, [(0, 1.0)]),),
        (Vectors.dense([1.0,4.0, 5.0, 0.0, 3.0]),),
        (Vectors.dense([1.0,6.0, 7.0, 0.0, 8.0]),),
        (Vectors.sparse(5, [(0, 9.0), (4, 1.0)]),)]
df = spark.createDataFrame(data, ["features"])

print(df.collect())

r1 = Correlation.corr(df, "features").head()
print("Pearson correlation matrix:\n" + str(r1[0]))

r2 = Correlation.corr(df, "features", "spearman").head()
print("Spearman correlation matrix:\n" + str(r2[0]))