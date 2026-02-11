
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
