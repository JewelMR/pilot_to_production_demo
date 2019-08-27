#This example walks through a logistic regression model using pyspark.
#We are predicting whether the client will subscribe (Yes/No) to a term deposit.
#Input variables: age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome.
#Output variable: deposit
#Links:
# https://www.kaggle.com/rouseguy/bankbalanced/version/1
# https://towardsdatascience.com/machine-learning-with-pyspark-and-mllib-solving-a-binary-classification-problem-96396065d2aa


#STEP ONE: IMPORT NECESSARY LIBRARIES AND LOAD DATA
from pyspark.sql import SparkSession
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName('ml-bank').getOrCreate()
df = spark.read.csv('C:/Users/MATSCJX/Desktop/bank.csv', header = True, inferSchema = True)
print("This is the schema of the data: ")
df.printSchema()

#STEP TWO: BRIEFLY EXPLORE DATA
#Target variable count
print("Here is the count for the target variable - deposit: ")
df.groupBy("deposit").count().sort("count",ascending=True).show()

#Summary of numeric features
print("Here is a summary of the numeric features: ")
numeric_features = [t[0] for t in df.dtypes if t[1] == 'int']
numeric_summary = df.select(numeric_features).describe().toPandas().transpose()
print(numeric_summary)

df = df.select('age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'deposit')
cols = df.columns

#STEP THREE: PREPARE DATA FOR MODEL BUILDING
    #Index each categorical column using the StringIndexer
    #Convert the indexed categories into one-hot encoded variables
    #Append binary vectors to the output at the end of each row
    #Use the StringIndexer again to encode our labels to label indices
    #Use the VectorAssembler to combine all the feature columns into a single vector column

categoricalColumns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
stages = []
for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]
label_stringIdx = StringIndexer(inputCol = 'deposit', outputCol = 'label')
stages += [label_stringIdx]
numericCols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

#STEP FOUR: CREATE PIPELINE TO CHAIN TRANSFORMERS AND ESTIMATORS
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(df)
df = pipelineModel.transform(df)
selectedCols = ['label', 'features'] + cols
df = df.select(selectedCols)

feature_df = pd.DataFrame(df.take(5), columns=df.columns).transpose()
print(feature_df)

#STEP FIVE: DIVIDE INTO TRAIN/TEST
train, test = df.randomSplit([0.7, 0.3], seed = 2018)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))

#STEP SIX: TRAIN CLASSIFIER - LOGISTIC REGRESSION
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
lrModel = lr.fit(train)

#Plot chart showing model coefficients
beta = np.sort(lrModel.coefficients)
plt.plot(beta)
plt.ylabel('Beta')
plt.title('Model Coefficients')
plt.show()

#Model Summary and plot chart showing ROC
trainingSummary = lrModel.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))

#Plot chart showing precision and recall
print("Precision and recall.")
pr = trainingSummary.pr.toPandas()
plt.plot(pr['recall'],pr['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()

#STEP SEVEN: TEST CLASSIFIER - LOGISTIC REGRESSION
print("Make predictions on the test set.")
predictions = lrModel.transform(test)
predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)

#STEP EIGHT: EVALUATE
evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(predictions))