
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

These methods also include [[Deleting data from DataFrame|deleting]] and [[Adding data to DataFrame]]
