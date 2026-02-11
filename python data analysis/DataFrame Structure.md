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
