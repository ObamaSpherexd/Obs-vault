#Library
NumPy is one of the most advanced Python libraries providing tools for complex math
[[how to run code in Obsidian]] - how to write and run code inside obsidian
# Arrays

### Comparison

python lists can contain different data types inside, can't handle vector operations without cycles => slow

NumPy arrays have only one data type, all operations can be performed on all items simultaneously

```python 

import numpy as np 
a = np.array([1, 2, 3, 4,5], float)

print('Array:', a)

print('Тип: ',type(a))

# обратное преобразование

print(a.tolist())
```


### Multidimensional arrays

![[Pasted image 20251121151451.png]]

First come rows then columns
```python 
import numpy as np
a = np.array([[1, -2, 3], [10, 5, 6]])

print(a)

print('1: ', a[0,0])

print('2: ', a[1,0])

print('3: ', a[0,1])

print('3: ', a[-1,-1])


print('4: ', a[1, :]) # second row fully

print('5: ', a[:, 2]) # third colimn fully

print('6: ', a[-1:, -2:]) # last row, second last column and to the end
print(len(a)) # 2 as there are only two elements, which are small arrays
print(a.shape) # prints amount of rows and columns
```

### Changing format

NumPy arrays can transform into different forms
```python
import numpy as np
a = np.array([1, 2, 3, 4, 5, 6])
print(a)
print(a.shape)
a=a.reshape(2,3) # i rows j columns
print(a) 
a=a.flatten() # return to 1-dimentinal array
print(a)

```

### Creating filled arrays

`numpy.arange()` returns ndarray, evenly spaced numbers
`numpy.arange(start, stop, step)`, unlike `range` can use float steps, by default start = 0, step = 1
#### [[Matrix]] Creation
NumPy has multiple functions to create filled matrixes, such as `numpy.ones()`, `numpy.zeros()`,`numpy.identity()` and others
`numpy.ones()` - create a matrix filled with ones
`numpy.ones(shape, dtype=float, order = 'C'`l, where shape is array size, dtype - data type, order - order of storing array in memory (C - strings, F - columns)
```python 
import numpy as np
print(np.ones((2,3))) # i rows j columns
print(np.zeros((3,2)))
print(np.identity(4)) # creates a n*n matrix with ones on main diagonal
print(np.eye(5,4,k=3)) # rows*columns willed with zeros, k= n of diagonal filled with ones
```
`numpy.zeros()` - create a [[Matrix]] filled with zeros, same syntax as `ones`
`numpy.identity()` 

### [[Math operations]]

**Main difference - can perform math operations on the items of the arrays themselves, not only on arrays**
```python 
import numpy as np
a_list = list(range(1, 7, 1))

b_list = list(range(6, 12, 1))

print('a: ', a_list)

print('b: ', b_list)
a_numpy = np.arange(1, 7, 1, dtype=int)

b_numpy = np.arange(6, 12, 1, dtype=int)

print('a: ', a_numpy)

print('b: ', b_numpy)

print(a_list+b_list)
print(a_numpy+b_numpy)
print(a_numpy-b_numpy)
# can't do with lists
print(b_numpy/a_numpy)
print(a_numpy % b_numpy)
print(b_numpy**a_numpy)
print(a_numpy//b_numpy)

print(np.sqrt(a_numpy))
print(np.floor(a)) # round to less
print(np.ceil(a)) # round to high
print(np.rint(a)) # round according to math


```

### [[Simple array operations]]

- 1-dimentional arrays
```python 
import numpy as np
a = np.arange(1, 6, 1)

print(a)

print('Сумма: ', a.sum())

print('Перемножение: ', a.prod())
print(a.mean()) # mean value, expected value
print(a.var()) # dispersion
print(a.std()) # standart deviation
print(a.argmin()) # number of the minimal item

a = np.array([6, 2, 5, -1, 0, 6, 2, 5, 4], float)

print(a.clip(0, 5)) # removes everything except values between given in brackets

a = np.array([5, 1, 1, 1, 1, 1, 1, 4])
print(np.unique(a)) # prints only unique values, simillar to set()
```

- n-dimentional arrays
  the main difference is that we need to use axes to define to which part of the array we refer
```python
import numpy as np
a = np.array([[5.2, 3.0, 4.5], [9.1, 0.1, 0.3]])
print(a)
print(a.min(axis=(0,1))) # in both rows and columns

a = np.array([[5, 2], [4, 1], [3, -1]])
print(a)
print('-'*30)
print(a.mean(axis=0)) # searches in rows, outputs as much values as there are rows
print('-'*30)
print(a.mean(axis=1)) # rearches in columns ...
print('-'*30)
print(a.mean())

```

### [[Logic operations with arrays]]

```python
import numpy as np
a = np.array([1, 3, 0, 4, 6])

b = np.array([0, 3, 2, 7, 8])
print(a == 4) # checks if any value in array fits, outputs array of T/F
a=np.array([1,3,0])
b=np.array([0,3,2])
print(a>10,type(a>b))
a=np.array([1,3,0,4])
print(a[a%2==0]) # checks requirements, fitting elements are put in an array

a=np.array([1,3,0])
mask=a>0
print(a[mask])
print(sum(mask),len(mask)) # only 2 values are valid, thus 1, but len is like array a
print(any(mask),all(mask)) # at least one is true / all are true

#logic

a=np.array([1,3,0])
print((a<3) * (a>0)) # both requirements are met
print((a<3) + (a>0)) # at least one is met

'''variants:
a[np.logical_and(a > 0, a < 3)]
a[a > 0 * a < 3]
'''
```

With `np.where(boolarray, truearray, falsearray)` we can create an array based on the task

```python 
import numpy as np
a = np.array([1, 3, 0, 4, 6, 9, 19])
print(a)
print(np.where(a != 0, 1 / a, a+1)) # condition a!=0 true: value is 1/a, false: a+1
print(np.where(a!=0)) # writes indexes of items fitting the requirements
print(a.argmax()) # prints the index of the first max element
```

One of the main features: we can select items by using other arrays
```python
import numpy as np
a = np.array([[6, 4], [5, 9]], float)
print(a >= 6) # prints T/F table
print(a[a>=6]) # prints 1-dimentional array consisting of values fitting the criteria
a = np.array([2, 4, 6, 8], float)
b = np.array([0, 0, 1, -2, 2], int)
print(a[b]) # prints values from a vith index of value b, 
'''
with n-dimentional we need
a=[...]
b=[...]
c=[...]
a[b,c]
'''
```

### [[Vector math]]

**Scalar multiplication** $a*b = \sum_{i=0}^{len(a)}a_i*b_i$
```python
import numpy as np
a = np.array([1, 2, 3], float)
b = np.array([0, 1, 1], float)
print(np.dot(a, b))
print(a @ b)
```

**Matrix multiplication**
```python
import numpy as np
a = np.array([[0, 1], [2, 3]], float)
b = np.array([2, 3], float)
d = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], float)
print(np.dot(b,a))
print(np.dot(a,b)) # or a@b
''' SHAPES HAVE TO BE ALIGNED!!! '''

```

Many [[Matrix]] math operations are presented in `linalg` module in numpy
```python 
import numpy as np
a=np.array([[0, 1], [2, 3]], float)
print(np.linalg.det(a))
```
