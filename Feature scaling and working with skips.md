#ML 
# Scaling
**feature scaling**
scaling - a common process of changing the range of a feature. it is a crucial step as features are measured in different units and therefore cover different scales.
age and salary columns have a very different scale.
this distorts results of such methods, as support vector method and k-nearest neighbors method, which need distances between measures. scaling helps avoid this problem. although linear regression and random forest methods don't need scaling, it is better not to overlook this stage when comparing different algorithms.
**2 common scaling methods**:
1. **Normalization**
   all values are between 0 and 1. discrete binary values are 0 or 1.
   calculation of a new value according to the formula:
   $$
   X_{norm}=\frac{X-X_{min}}{X_{max}-X_{min}}
   $$
2. **Standardization**
   scales values including standard deviation. if standard deviation of different functions is different, the scale also differs. this lowers the importance of outbursts. $\sigma$ - standard deviation, $\mu$ - mean value
   $$
   X_{std}=\frac{X-\mu}{\sigma}
   $$
   standardization takes all initial values and no matter the staring deviation and units, transforms them into a set of dispersion values with **zero mean value** and **singular standard deviation** (to normal distribution)
`sklearn.preprocessing` has a lot of different functions of preprocessing, as well as functions which can scale features:
- **StandardScaler()** - class for standardization
- **MinMaxScaler()** - class for normalization
```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd

# features vector
X_train=np.array(([1000.,-15.,2.],[2500.,10.,1.],[3000.,13.,-1.]))

X_train = pd.DataFrame(X_train,columns=['f1','f2','f3'])
X_train
```
## Standardization example:
```python
std=StandardScaler()

X_std=std.fit_transform(X_train)
std.fit(X_train)
X_std=std.transform(X_train)
X_std_test=std.transform(X_test)
X_std=pd.DataFrame(X_std,columns=['f1','f2','f3'])
X_std

X_std['new']=[1000,1001,1002]
X_std

X_std['new']=std.fit_transform(X_std[['new']])
X_std
  
# mean for every column
X_std.mean(axis=0).round(0)

# mean deviation for every column
X_std.std(axis=0).round()
```
## Normalization example
```python
mmsc=MinMaxScaler((-1,1))
X_norm=mmsc.fit_transform(X_train) # type - numpy array
X_norm=pd.DataFrame(X_norm,columns=['f1','f2','f3'])

# mean for every column
X_norm.mean(axis=0)

# mean deviation for every column
X_norm.std(axis=0)
```
additionally:
![[Pasted image 20260207172757.png]]
# Working with skips
previously we have only focused on creating and fitting a model, but every task in ML is linked with 2 stages:
- working with data
- working with ML algorithms
as quality of data influences the effectivity of algorithms of ML, before creating and fitting algorithms it is important to focus on EDA of data and its preprocessing.
one of the most common problems is missing values. skips can happen due to various causes, such as errors while collecting data, technical difficulties or human error.
processing skips is important, as they may:
- disrupt algorithms work
- lower accuracy
- include bias to analysis
- many models can't work with skips at all
skips are usually represented as NaN or None values. or sometimes they can be encoded in special values, such as  -1, 999, ?, NoneType and others

2 main strategies:
- deleting object or features with skips
- filling skips:
	- filling them with mean/median/mode on the whole feature
	- filling with mean/median/mode on the aggregated feature
	- filling using ML model
	- filling with previous or next row (using ffill or bfill methods)
	- changing using interpolation
	- using premade algorithms based on ML: `SimpleImpputer()`, `KNNInputer()`;
	- other
looking into these strategies using a Kaggle dataset:
we need to use API token (go into settings of the account and generate an API token)
![[Pasted image 20260207180703.png]]
```python
import pandas as pd
df=pd.read_csv('...')
df.isnull().sum() # counting null elements
df_nan=df.loc[:,df.isnull().any()]
# percent of missing values
df_nan.isnull().sum()/len(df_nan)*100
```
we can clearly see that there are columns where more than 80% blank spaces. these variables are best deleted, as filling them may only worsen the result.
`df=df.drop(['Alley','PoolQC','Fence','MiscFeature','FireplaceQu'],axis=1)`
filling one of variables with 0:
`df['MasVnrArea']=df['MasVnrArea'].fillna(0)`
filling other variable with mean of the whole variable/median
```python
# using mean
m=df['LotFrontage'].mean()
df['LotFrontage'].fillna(m,inplace=True)
# df['LotFrontage']=df['LotFrontage].fillna(m)

# using median
```
filling using mode for non-integer variables
```python
# using mode
df['GarageFinish'].fillna(df['GarageFinish'].mode()[0],inplace=True)
```
filling using previous or next data:
```python
df['GarageType']=df['GarageType'].fillna(method='ffill')
df['Electrical']=df['Electrical'].fillna(method='bfill')
```
filling using interpolation:
```python
df['GarageYrBlt'].interpolate(method='linear',direction'forward',inplace=True)
```
we've looked into the simplest methods of working with skips, but we need to understand that the method of processing skips depends on the task and possible influence of skips on results. we need to thoroughly analyze data, test different approaches and understand their usefulness