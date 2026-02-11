`numpy.ones()`, `numpy.zeros()`,`numpy.identity()` and others
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
