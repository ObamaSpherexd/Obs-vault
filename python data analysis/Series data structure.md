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
