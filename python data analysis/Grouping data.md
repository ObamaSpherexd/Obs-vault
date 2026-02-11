One of the basis functions of data analysis is grouping and aggregation.
![[Pasted image 20251205150510.png]]
```python
df=pd.read_csv('test.csv')
print(df.head())
tmp=df.groupby(['Gender','Class']).agg({'Age':['mean','max']}) # sorts by quality in order
print(df.groupby(['Gender']).agg({'Age':['mean','max']}))
print(tmp.reset_index())
tmp=df.groupby(['Gender']).agg({'Age':['mean'],'Flight Distance':['mean','max']})
print(tmp)

```
![[Pasted image 20251208134341.png]]
Pandas has a DataFrame.pivot_table function, which allows to quickly turn DataFrame into a table
```python
df['Class'].value_counts()
df.pivot_table(index=['Gender'], values='Age', aggfunc=['mean'])
```
### Unique values
```python
df['Gender'].unique() # outputs unique values
df['Gender'].value_counts()
```
### `loc` and `iloc` methods
**DataFrame.loc[]** - access a group of rows or columns by tag or logical array
Acceptable inputs:
- Singular tag
- Array of tags
- A split 
**DataFrame.iloc[]** - access values by integer indexes
Acceptable inputs:
* Integer
* Array of integers
* Split
```python
print(df.loc[df['Gender'==0],'Age'].mean())
print(df.loc[df['Gender'==1],'Age'].mean())
print(df.groupby(['Gender']).agg({'Age':'mean'}))
print(df.loc[(df['Gender']==1) & (df['Class']=='Business')])
```
### Apply method
Some data can be changed while analysing it. We can operate with either certain values or whole rows/columns. Last method is more preferrable as it boosts productivity while operating large quantities of data.
```python
def increase_age(x):
	return x+1
df['Age']=df['Age'].apply(increase_age)
# df ['Age'] = df['Age'].apply(lambda x:x+1)
# df['Age'] = df['Age']+1
```
### Lambda functions
```python 
def my_func(x):
	x=x*2
	return x
```
Lambda-functions are anonymous (nameless). `lambda args:expression`
```python 
double=lambda x:x*2
print(double)
# equivalent to
def double(x):
	return x*2

```
### Lambda-functions and higher level functions
We use lambda functions when we need to use a nameless function for a short period of time. We frequently use them as an argument of a higher level function (function which takes other functions as arguments). They are frequently used with such functions as `filter()`, `map()`, `reduce()`, `apply()` and others.
```python
d={0:'Man',1:'Woman'}
df['Gender'] = df['Gender'].map(d)
df.head()
df['Age'] = df['Age'].apply(lambda x:x+1)
'''equivalent
for index,row in df.iterrows():
	df.loc[index,'Age']=row['Age']+1
or
df['Age'] = df['Age']+1'''
```
