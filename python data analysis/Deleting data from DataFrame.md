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
