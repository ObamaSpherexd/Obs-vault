#Library 
Pandas is a high-level Python library for data analysis
Pandas is the most advanced and fast-developing library for processing data

Pandas introduces new data structures - **Series** and **DataFrame**

It is used by running the command `import pandas as pd`

## [[Series data structure]]

**Series** - one-dimensional arrays, similar to lists, but operations are done on the whole list, while in a Series data can be manipulated by items,

A Series constructor looks like this:
`pd.Series(data=None, index=None, dtype = None, name=None, copy=False, fastpath=False`
- **data** - array, dict or scalar value, on which the Series will be built
- **index** - list of tags, which will be used to access Series elements, length of the list should be the same as **data** length
- dtype - data type
- copy - creates a copy of array if True

### Creating Series from array

```python
lst=[1,2,3,4,5]
arr=np.array(lst)
s=pd.Series(data=lst)
# if the index is not given, pandas automatically creates RangeIndex from 1 to N-1, where N is the total amount of elements
type(s) # pandas.core.series.Series

lst=['Marie','Pete','Mark']
s=pd.Series(data=lst,index=['name','age','smt'])
# creating Series with tags a,b,c
#name Maria
#age Pete
#smt Mark
```

```python
import numpy as np
ndarr=np.array([1,2,3,4,5])
type(ndarr) # numpy.ndarray
s2=pd.Series(ndarr,['a','b','c','d','e'])
```

```python
d={'a':1,'b':2,'c':3,'a':9}
s3=pd.Series([1,5,6,3],['a','c','d','a'])
print(s3)
```

```python 
a=7
s4=pd.Series(6,['a','b','c','a','a'])
print(s4)
```

Pandas has attributes to get a list of elements and their indexes - **values*** and **index** respectfully
```python
print(s4.index)
print(s4.values)
print(s4.shape)
```
### [[Working with Series elements]]

Series elements can be called by index like a regular list

```python
l=[1,5,6,7]
print(l[0],l[-1],l[1:3])
s5=pd.Series(['Ian','Pete','Marie','Nastia','Nate'], ['a','b','c','d','a'])
print(s5)

print(s5[2])
print(s5['c'])
print(s5[-3])
print('---------')
print(s5['a'])
```

We can use tags so that working with **Series** will be similar to working with dicts
```python
s5=pd.Series(['Иван', 'Петр', 'Мария', 'Анастасия', 'Федор', 'Надя'], [1, 2, 4, 2, 6, 6])
print(s5)
print(s5[6])
# print(s5[-1]) will not work
```

*We can access Series elements using [ ] by either:*
*1. Giving an index*
*2. Giving a named tag if there are any*

#### Accessing elements by given conditions

We can put a condition in brackets, the result will be all elements fitting the criteria.
Any logical operation returns `True/False` and then the program only takes indexes with `True` value
```python 
s6=pd.Series([0,-13,1,-5,0],['a','b','c','d','e'])
print(s6[s6==0])
print(s6==0) # mask
l=[10,0,13,1,5,0]
for i in l:
	if i==0:
		print(i)
print('------------')
print(s6[['a','c','e']]) # returns elements with given tags

print('----------------------------------')
s6=pd.Series([10,13,1,5,0,6,4,8],['a','b','c','b','c','f','g','h'])
print(s6)
# type(s6[0]) is list
print(s6['a'::2])

```

### [[Working with Series]]

Series structure behave like vectors and can be added, multiplied by a number etc
```python
s7=pd.Series([10,2,30,40,50,8],['a','b','c','d','e','a'])
print(s7)
s7=s7**2
print(s7)
''' List - slow and bad
Numpy array - fast and good
pandas array - just good
'''
l1=[1,2,3]
l2=[4,5,6]

s7=pd.Series([10,20,30,40,50],['a','a','c','d','e'])
s8=pd.Series([1,1,1,1,0],['a','b','c','h','d'])
print(s7+s8) # for proper addition their len should be =
# only elelents with the same tag are operated
l=[1,2,4]
print(l*3)
print(s7*3)
```

## [[DataFrame Structure]]

We can call DataFrame column by either:
- `df.column_name`
- `df['column_name']`

```python
nda=np.array([[1,2,3],[10,20,30]])
df=pd.DataFrame(nda)
print(df)

d=[{'name':'victor','age':18},
	{'name':'marie','age':21},
	{'name':'ivan','age':19}]
df=pd.DataFrame(d)
print(df)
print('='*50)
print(df['name'])
print(df.name)
```

There are several ways to work with rows and columns:
==**DataFrame.loc[]**== - access a group of rows/columns (only one of them) by tags of logical array, accepts:
- singular tag like 5 or 'a'
- list or array of tags
- tag slices
==**DataFrame.iloc[]**== - allows accessing DataFrame elements by integer tags
- whole numbers
- array or list of numbers
- number slices (1:7)
```python
print(df)
print(df.iloc[-1])
print(df.iloc[2,1])
print('='*50)
print(df.iloc[0:,0:-1:1])
print(df.iloc[0,1]) # intersection of the 0 row and 1 column
print(df.iloc[:,1]) # all values of 1 column

print(df.loc[:,'name']) # all items on column name for all rows
df.iloc[:,-1]=-100
print(df)
```

.loc[] and .iloc[] are also applicable for Series structure
We can also write logical cases to filter values
```python
d = [{"Name": "Виктор", "Age": 18, 'Пол': 'м'},{"Name": "Мария", "Age": 21, 'Пол': 'ж'},{"Name": "Иван", "Age": 19, 'Пол': 'м'},{"Name": "Надя", "Age": 22, 'Пол': 'ж'},{"Name": "Варя", "Age": 18, 'Пол': 'ж'},{"Name": "Максим", "Age": 23, 'Пол': 'м'}]

df = pd.DataFrame(d)
print(df)

print(df[df['Age']>=20])
print(df.loc[(df['Пол'] == 'ж') & (df['Age'] >= 20), ['Name', 'Age']])
print('='*50)
df = pd.DataFrame({
    'country': ['Kazakhstan', 'Russia', 'Belarus', 'Ukraine'],
    'population': [17.04, 143.5, 9.5, 45.5],
    'square': [2724902, 17125191, 207600, 603628]},
                  index=['KZ', 'RU', 'BY', 'UA'])
print(df)
print(df[df.population>10]['country'])

df['new_col']=df['square']/df['population']
print(df)
```

### Easier access to values
**We can access values and select them using loops**
One of these approaches is using *.iterrows()*. This method returns an iterator which generates an index for each row and the data in it in a *Series* format
```python
d = [{"Name": "Виктор", "Age": 18},
     {"Name": "Мария", "Age": 21},
     {"Name": "Иван", "Age": 19}]

df = pd.DataFrame(d)

for index, row in df.iterrows():
    print(row)
    print('_________________________')
    row['Name'] = row['Name'].upper() # does NOT change df
    df.loc[index, 'Name'] = row['Name'].upper() # Does change df

print(df)

```
The second approach is using *.itertuples()*. It returns an iterator which generates a named tuple with index of row and data
```python
for row in df.itertuples():
	print(row.Name, row.Age)
```
In this case row is a named tuple, we can call cell values through named tuple attributes, which correspond to column names.
**Both these methods are used to iterate DataFrame strings, but they cannot change values**. If we need to change data, we should use other approaches like vectorized operations.

```python
for index,row in df.iterrows():
	df.loc[index,'Name'] = row['Name']+ ' Hi'
print(df)
```

### [[DataFrame methods and attributes]]
```python
d = [{"Name": "Виктор", "Age": 18},
     {"Name": "Мария", "Age": 21},
     {"Name": "Иван", "Age": 19},
     {"Name": "Петя", "Age": 34}]

df = pd.DataFrame(d)
print(df)

print(df.loc[0]) # returns first row
print(df.shape)
print(df.index) # returns DataFrame indexes for rows
print(df.columns) # returns column names
print(df.head(2)) # returns first 2 rows
print(df.tail(2)) # returns last 2 rows
print(df.info) # returns full information about this DataFrame

```
Methods *.isna()* and *.isnull()* return 2 DataFrame objects, where True is for NaN and False - for everything else
```python
d = [{"Name": "Виктор", "Age": 18},
     {"Name": "Мария", "Age": np.nan},
     {"Name": "Иван", "Age": 19},
     {"Name": "Петя",}]

df = pd.DataFrame(d)
print(df.isnull().sum())
print(df[df['Age'].isnull()==False]) # returns all rows with data in age
d = [{"Name": "Виктор", "Age": 18},
     {"Name": "Мария", "Age": np.nan},
     {"Name": "Иван", "Age": 19},
     {"Name": "Иван", "Age": 25}]

df = pd.DataFrame(d)
print(df['Name'].unique(), df['Name'].nunique()) # returns unique values and their count
print(df['Name'].value_counts()) # returns amount of every value in column
```

#### [[Deleting data from DataFrame]]
One of methods of deleting rows or columns from DataFrame:
`DataFrame.drop(labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise'`
- Labels - tags of index or column to be deleted
- axis - {0 or 'index', 1 or 'columns'}
- index - =axis=0
- columns - =axis=1
- level - for MultiIndex - level following which tags will be deleted
- inplace - if False will return a DataFrame copy, if true will rewrite changes in the initial DataFrame
```python
d = [{"Name": "Виктор", "Age": 18},
     {"Name": "Мария", "Age": np.nan},
     {"Name": "Иван", "Age": 19},
     {"Name": "Иван", "Age": 25}]

df = pd.DataFrame(d)
print(df)
print(df.drop(columns=['Age'])) # THIS STRUCTURE DOES NOT CHANGE THE INITIAL DATAFRAME
df.drop(columns=['Age'],inplace=True) # THIS DOES
print(df)
print(df.drop(2,axis=0))
print(df.drop(0,axis=0))
df=pd.DataFrame(d)

print(df.drop([0,1])) # rows 0 and 1 deleted

```
There is also a `dropna()` method, which deletes rows or columns with null values
`DataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)`
- axis - {0 or 'index', 1 or 'columns'}, 0 by default. Determines whether the rows or the columns are being deleted.
- how - determines whether a row/col is deleted from DataFrame when we have at least one NA ('any') or if all values are NA ('all')
```python
d = [{"Name": "Виктор", "Age": 18},
     {"Name": "Мария", "Age": np.nan},
     {"Name": "Иван", "Age": 19},
     {"Name": "Иван", "Age": 25},
     {"Name": 'Надя', "Age": np.nan}]

df = pd.DataFrame(d)
print(df.dropna(axis=1))
print(df.dropna(axis=0,how='any')) # all the rows with nan value will be dropped
```
#### [[Adding data to DataFrame]]
We can add a new column to an existing DataFrame using this construction:
`df['new_column'] = values`
- values - data in new column (numbers, array, Series)
```python
d = [{"Name": "Виктор", "Age": 18},
     {"Name": "Мария", "Age": 21},
     {"Name": "Иван", "Age": 19},
     {"Name": "Иван", "Age": 25},
     {"Name": 'Надя', "Age": 20}]

df = pd.DataFrame(d)
print(df)
df['University'] = ['NRNU MEPhI', 'MIPT', 'MITP', 'NRNU MEPhI', 'BMSTU']
print(df)
df['level'] = 'bachelor'
```
There is also a method allowing us to join Series and DataFrame objects
`pandas.concat(objs, axis=0, join='outer', ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=False, copy=True`
- objs - sequence or comparison of Series and DataFrame objects
- axis - 0 by default, merging axis (by axis or columns)
```python
s1=pd.Series(['a','b'])
s2=pd.Series(['c','d'])
print(pd.concat([s1,s2])) # indexes are presented as they were initially distributed
print(pd.concat([s1,s2], ignore_index=True))

df1 = pd.DataFrame([['Петр', 19], ['Иван', 22]],columns=['Name', 'Age'])
print(df1)

df2 = pd.DataFrame([['Мария', 20], ['Анастасия', 18]], columns=['Name', 'Age'])
print(df2)
tmp=pd.concat([df1,df2],axis=0)
print(tmp)
print(df.iloc[df['Age']>20,5:15])

```

## [[Data import and Export]]
We can use .csv or .xlsx files to create DataFrames
**CSV** (Comma-Separated Values) - text format for presenting data in table.
Each row is distinct from another, columns are separated by special simbols like a comma.
To read .csv files we use `read_csv()`
* filepath_or_buffer - path to file 
* sep - used separator
![[09.ods]]
```python 
df=pd.read_excel('09.ods',header=None)
print(df.head())
print(df.shape()) # (16000,6)
df2=dp.read_csv('File.csv')

df2.max() # =99 maximum value in a cell
df2.sum() # sum of all values in a row
df2.max(axis=0) # finds max in every column, be default axis=1
df2.mean(axis=0) # mean in every column
df.duplicated() # checks if a value if duplicated False/True
# find dublicates in all columns
duplicateRows=df[df.duplicated()]
duplicateRows=df[df.duplicated(['col1','col2'])] # finds dupes in particular columns
print(df.columns) # prints names of columns
```
### Data export
DataFrame and Series can be saved as .csv and .xlsx files
`DataFrame.to_csv(path_or_buf=None, sep=',', na_rep='', float_format=None, columns=None, header=True, index=True, index_label=None, mode='w', encoding=None, compression='infer', quoting=None, quotechar='"', line_terminator=None, chunksize=None, date_format=None, doublequote=True, escapechar=None, decimal='.', errors='strict', storage_options=None)`
* path_or_buf - path to file
* sep - line with len1, divider for outcoming file
To write a singular object to Excel .xlsx, we need to inly give the end file name. To write in multiple lists we need to create an ExcelWriter object with the name of the end file and point out the list in output file.
`DataFrame.to_excel(excel_writer, sheet_name='Sheet1', na_rep='', float_format=None, columns=None, header=True, index=True, index_label=None, startrow=0, startcol=0, engine=None, merge_cells=True, encoding=None, inf_rep='inf', verbose=True, freeze_panes=None, storage_options=None)`
## [[Grouping data]]
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
### `loc` And `iloc` methods
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
Some data can be changed while analysing it. We can operate with either certain values or whole rows/columns. Last method is more preferable as it boosts productivity while operating large quantities of data.
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

## [[Multiple category processing, separating data in columns]]

**CODE WITH OUTPUTS IS IN COLAB:** https://colab.research.google.com/drive/1zOaZcjLdEI0MPV-sJVSkRUFVfmGY2b1n?usp=sharing


`pandas.Series.str.split` - splits the string around the given separator
Series.str.split(_pat=None_, _*_, _n=-1_, _expand=False_, _regex=None_)  regex determines whether there are regular expressions
```python
s='Hello world word'
print(s.split('w'))
print(s.replace('wo','v'))
tmp=df['topics'].str.replace(', ','/') # replaces , with / in list of topics
tmp=tmp.str.split('/',expand=True) # splits by /, extends rows to needed max len
print(tmp.iloc[:,0].unique)

print(tmp.iloc[:,1].unique)

print(tmp.iloc[:,2].unique) # print unique values in each column after splitting
tmp2=df['topics'].str.split(r'[\w,][,][\s]+',regex=True,expand=True) # deletes last letter if it is s then splits
tmp3=df['topics'].str.get_dummies(',') # filters by criteria and puts a number of its calls
df['topics']=df['topics'].str.replace(', ',',')

tmp4=df['topics'].str.get_dummies(',') # same as last

```

## [[Regular expressions]]
**Regular expressions** - blueprints which are used to compare symbols in strings

Repos of most common regular expressions:
- [https://regex101.com](https://www.google.com/url?q=https%3A%2F%2Fregex101.com)
- [http://www.regexlib.com](https://www.google.com/url?q=http%3A%2F%2Fwww.regexlib.com)
- [https://www.regular-expressions.info](https://www.google.com/url?q=https%3A%2F%2Fwww.regular-expressions.info)
```python
import re
df=pd.read_csv('data_regex.csv')
```
Regular expressions usually contain special symbols which are called *metasymbols*: `[] {} () \ * + ^ $ ? . |`
Every predetermined symbol class starts from `\`, each class is the same with a symbol from a set
- `\b` - any number `0-9`
- `\D` - anything but a number
- `\s` - any space-symbol (tab, space, \n)
- `\S` - anything but a space
- `\w` - any word symbol (letter, number, _)
- `\W` - anything but a word symbol
```python
# check that screen resolution has 4 numbers
if re.fullmatch(r'\d{4}','1920'):
	print('yes')
else:
	print('no')

if re.fullmatch(r'\d{4}','768 '):
	print('yes')
else:
	print('no')
	
```
*Symbol class* - sequence in a regex, coinciding with 1 symbol. In order for the coincidence to be composed of several symbols, we need to specify a quantificator after the class
Quantificator {4} repeats `\d` 4 times 
`[]` define users symbol class, coinciding with one symbol. So `[aeiou]` = lower register vowel. `[A-Z]` - upper, `[a-z]` - lower, `[a-zA-Z]` - any letter
Let's check the name - the sequence starts with Upper register, followed by any amount of lower register:
```python
if re.fullmatch('[A-Z]*','Full HD 1920x1080'):
	print('yes')
else:
	print('no')

if re.fullmatch('[A-Z]*','FULLHD'):
	print('yes')
else:
	print('no')
```
Metasymbols in users symbol class are interpreted as literal symbols, so `[*+$]` means `+` `*` or `$`
Quantificators `*` and `+` are maximums - they coincide with the max possible amount of symbols.
`{n,}` coincides with at least n values
```python
if re.fullmatch(r'\d{3,}','Full HD 1920x1080':
	print('yes')
else:
	print('no')
```
## [[Functions of the `re` module]]
```python
re.match()
re.search()
re.findall()
re.split()
re.sub()
re.compile()
```
### Search function
This function searches for the first `pattern` in `string` with additional `flags`:
`re.search(pattern,string,flags=0)`
Returns `match` object if there is any, else `None`
`match` conducts search in the beginning of the string, `search` - in the whole string
`search()` searches in the whole string, but returns only the first fit
```python
re.search(r'FULL', 'FULL HD 1920x1080 FULL') # <re.Match object; span=(0, 4), match='FULL'>
re.search(r'(\d+)\D(\d+)', 'IPS Panel Full HD / Touchscreen 1920x1080') # <re.Match object; span=(32, 41), match='1920x1080'>
```

### Findall function
Returns all found matches, no constraints.
`re.findall(pattern,string,flags=0)`
```python
re.findall(r'[0-9]*[x][0-9]*', '1920x1080 IPS Panel Touchscreen / 4K Ultra HD 3840x2160')
# ['1920x1080', '3840x2160']

def get_formatted_screen(text):
	retult=re.findall(r'[0-9]*[x][0-9]*',text)
	return result[0]
df['ScreenResolution'].map(get_formatted_screen)
```

### Split function
Splits the string according to the pattern
`re.split(pattern, string, [maxsplit=0])`
```python
re.split(r'[\s]', '1920x1080 IPS Panel Touchscreen / 4K Ultra HD 3840x2160') # split by space

```

### Sub function
Searches for a pattern and replaces it with an inputted value, if the pattern is not found then no changes.
`re.sub(pattern,repl,string)`
```python
re.sub(r'GB','','16GB')
def get_formatted_ram(value):
	value=re.sub(r'GB','',value)
	return value
df['Ram']-df['Ram'].map(get_formatted_ram)
```

### Changing Data type in DataFrame
```python
df['Ram']=df['Ram'].astype(int)
```

## [[Dynamic change of data type in columns]]
If a DataFrame has a lot of columns and we don't know which do we need to transform, we can use `apply()`, which applies the inputted function to each column
```python
df=df.apply[lambda col: col.astype(float) if col.str.replace('.','').str.isdigit().all() else col]
```
Here we use a function which checks if all rows in a column have only numbers, and if so, it makes a column into numbers with floating point

### Changing name of columns
```python
import pandas as pd

df=pd.read_csv('datadset.csv')
tmp=df['topics'].str.split(',',expand=True)

tmp.columns=['col1','col2','col3','col4','col5']

tmp=tmp.rename(columns={'col1':'col1_renamed'})

```

## Additional
```python
df=pd.read_csv('data_regex.csv')

resolution-'IPS Panel 4K Ultra HD 3840x2160'
width_height=resolution.split('x')
# width=int(width_height[0].strip())
# height=int(width_height[1].strip())

df['ScreenResolution'].str.replace(r'[\D]',' ',regex=True)

A={8,2,11,4,42,6}
print(A.pop()) # prints sorted from 4 up, first one (2) is popped

df['ScreeenResolution'].str.extract(r'(?<=x)([0-9]+)').astype(float)

```
