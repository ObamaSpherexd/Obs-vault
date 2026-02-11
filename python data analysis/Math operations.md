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
