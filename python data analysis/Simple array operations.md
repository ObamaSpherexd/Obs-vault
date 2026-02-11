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
