```python
import micropip 
await micropip.install('numpy') 
import numpy as np 
a = np.random.rand(3,2) 
b = np.random.rand(2,5) 
print(a@b)
await micropip.install('matplotlib')
import matplotlib.pyplot as plt
await micropip.install('pandas')
import pandas as pd
await micropip.install('seaborn')
import seaborn as sns

```
