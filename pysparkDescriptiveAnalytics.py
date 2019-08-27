#This example walks through a descriptive analytics example using PySpark - SQL.

#STEP ONE: IMPORT NECESSARY LIBRARIES AND LOAD DATA
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark import SparkFiles
sc = SparkContext()
sqlContext = SQLContext(sc)

#LOADING DATA USING GITHUB DESTINATION
url = "https://raw.githubusercontent.com/guru99-edu/R-Programming/master/adult_data.csv"
sc.addFile(url)
df = sqlContext.read.csv(SparkFiles.get("adult_data.csv"), header=True, inferSchema= True)

#STEP TWO: EXPLORE DATA
#print the schema
print("This is the schema of the data:")
df.printSchema()

#show first 5 rows
print("This shows the first five rows of the data:")
df.show(5, truncate = False)

#select certain columns
print("You can select specific columns:")
df.select('age','fnlwgt').show(5)

#count by group
print("You can group by education and count the records in the group:")
df.groupBy("education").count().sort("count",ascending=True).show()

#describe dataset - gives simple stats
print("The describe function provides baseline descriptive analytics.")
df.describe().show()

#describe one column
print("There is the option to describe just one column.")
df.describe('capital-gain').show()

#describe subset of data
print("There is the ability to describe just a subset of the data: describe records where age is equal to 81.")
df.filter(df.age==81).show()

#group data and compute statistical operations: mean
print("This is an example of grouping and then computing the mean.")
df.groupby('marital-status').agg({'capital-gain': 'mean'}).show()