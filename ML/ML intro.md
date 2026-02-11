#ML
**machine learning** - class of methods of automatic creation of prediction models based on data.
it is a subcategory of AI and data science, specialising in using data and algorithms to imitate experience gathering with a gradual accuracy increase.
**goal of ML** - predict the result based on the input data. the broader they are, the easier it is to find patterns and the better the result.
such as:
- speech detection
- computer vision
- medical diagnosis
it can be used in medical area to analyse the input data, combine them to create a prediction of an illness progression, to plan treatment and to monitor.
- feed recommendations
- stock market predictions
- house price predictions
**AI** - name of the whole area of science
**Machine learning** - section of AI
**Neural networks** - a type of ML
**Deep learning** - neural networks architecture, one of approaches for their creation and teaching
## Key concepts of ML
1. **Data** (dataset)
we want to detect spam - we need examples of letters, want to predict the market - need history. we need as much data as possible, at least tens of k.
they are collected in all ways possible, manually (long, no errors), automatically, using other people.
there is a constant hunt for good data. the better the dataset, the more probable the success.
2. **Features**
features - car mileage, sex, price etc.
algorithm has to know what does it need to find. proper feature collection takes more time than all the other teaching.
3. **Target**
the answer, the unknown variable that we want to find/predict.

features are usually X, target is Y, so any algorithm tries to find and build a dependence function $Y=f(X)$
example:
```python
from google.colab import files
uploaded = files.upload()
import pandas as pd

df = pd.read_csv("USA_cars_datasets (2).csv")

df.head(10)
# y - target vector[y1]
# x = [x1,x2,..] - features vector (x1 = 0, x2 = 1)
# y = a1*x1 + a2*x2 ...
```
4. **Algorithm**
one task can be resolved by various methods almost anytime. accuracy, speed and the model size depend on the approach, but if the data is bad, no algorithm can help.
![[Pasted image 20260205111027.png]]
5. **Model**
A model can be a mathematical interpretation of a real event. to generate a ML model we'll need to give data to the ML algorithm for training.
6. **Training**
in the process of training we use an algorithm with a dataset. the algorithm finds patterns in the data so that input data corresponds to the goal. the result is an ML model which can be used for prediction.
## Types of ML
![[Pasted image 20260205112128.png]]
![[Pasted image 20260205112142.png]]
### Supervised learning
the most popular method is supervise learning. we have a certain amount of tagged data for which we have a proper answer and prediction. we try to teach our model on this data and then create a prediction on new data which we have not seen previously.
***Types of tasks***:
1. regression - prediction based on a dataset with different features. in the output we should have a number (house price, stock price, revenue). Linear or Polynomial regression
2. classification - we get a categorical answer based on a set of features. has a finite amount of answers (yes/no). Naive Bayes, decision tree, logical regression etc.
### Unsupervised learning
doesn't need tagged data. we just have data, and we try to use the dependencies between them and make conclusions based on the data or transform it. we can do so as this data has some kinds of natural dependencies we can use.
***Types of tasks***:
1. Dimension reduction
a classical task. we have a dataset with a lot of features and our model wants to study on 100 most important features, not 10000.
this way we would be able to brush off rudiment features saving peak productivity. so we want to lower dimensions by keeping only the important features. we also can generate new features or model a new dimension.
2. task of clustering
we have a lot of data and we try to find clusters or groups of similar objects.
using clustering methods we can automatically group similar objects, find anomalies, isolated objects, which require additional studying or removal, clusterization allows to conduct a more detailed analysis.
3. association rules
association rule learning (ARL) - a simple but frequently used method of finding links in datasets
General case: "who bought X also bought Y". Base - transaction analysis with a unique itemset from items. using ARL we can find the rules of coincidences in items inside one transaction, which are then sorted by force.
4. recommendations systems
## What is what
### Dataset 1
https://www.kaggle.com/uciml/german-credit
```python
from google.colab import files
uploaded = files.upload()
import pandas as pd

df = pd.read_csv("german_credit_data (3).csv")

df.head(10)

# 1. how long is the credit perioud
#  Y = Duration

# 2. determine sex Y = Sex

# 3. devide into simillar groups
```
1. task: find purpouse
2. predict loan price
3. predict sex
4. predict age
5. group people by similarity
### Dataset 2
https://www.kaggle.com/kaushiksuresh147/customer-segmentation
```python
from google.colab import files
uploaded = files.upload()
import pandas as pd

df = pd.read_csv("Train (9).csv")

df.head(10)
```
6. determine segmentation of the buyer
7. determine profession by features
### Dataset 3
https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python
```python
from google.colab import files
uploaded = files.upload()
import pandas as pd

df = pd.read_csv("Mall_Customers (1).csv")

df.head(10)
```
8. find a group of the most promising clients, plan a involvement strategy (find groups of customers, select the most suitable)
### Dataset 4
```python
from google.colab import files
uploaded = files.upload()
import pandas as pd

df = pd.read_csv("News_dataset.csv", sep=';')

df.head(10)
```
9. determine news type by text
### Dataset 5
https://www.kaggle.com/mirichoi0218/insurance
```python
from google.colab import files
uploaded = files.upload()
import pandas as pd

df = pd.read_csv("insurance (1).csv")

df.head(10)
```