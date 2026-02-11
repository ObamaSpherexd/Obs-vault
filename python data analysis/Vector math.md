
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
