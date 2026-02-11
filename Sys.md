Python library used to work with system Inputs
`sys.stdin` - standard input, checks what information has been given
Example:
```python
import numpy as np
import sys

data=np.loadtxt(sys.stdin,dtype=int) # .loadtxt loads data from a txt file
zeros=np.sum(np.all(data==0, axis=0))
print(zeros)
```
