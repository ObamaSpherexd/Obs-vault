If a DataFrame has a lot of columns and we don't know which do we need to transform, we can use `apply()`, which applies the inputted function to each column
```python
df=df.apply[lambda col: col.astype(float) if col.str.replace('.','').str.isdigit().all() else col]
```
Here we use a function which checks if all rows in a column have only numbers, and if so, it makes a column into numbers with floating point

### Changing name of columns
```python
import pandas as pd

df=pd.read_csv('datadset.csv')
tmp=df['topics'].str.split(',',expand=True)

tmp.columns=['col1','col2','col3','col4','col5']

tmp=tmp.rename(columns={'col1':'col1_renamed'})

```
