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

