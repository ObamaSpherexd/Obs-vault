# Simple analysis of the selected elements
#Statistics 
## Parameter evaluation
**Every selection is values of a random magnitude**
By analyzing the selection, we can:
- Evaluate the parameters of a random value
- Check hypotheses about values of the said parameters, their common characteristics
- Create confidence intervals for values
For point estimates of random values various statistics are used
**Statistic** - any function from the selection
Example: hockey players dataset https://habr.com/post/301340/:

```python
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv('drive/MyDrive/hockey_players.csv', encoding="ISO-8859-1")
df.head()
```
Let's say we have a set $X=(x_{1},x_{2},\dots,x_{n})$ of values of a 1d random variable
One of the most common and easiest statistic is **arithmetic mean** $\overline{X}$
$$
\overline X = \frac{1}{n}\displaystyle\sum_{i = 1}^n x_i
$$
This is an evaluation of mathematical expectation.
This means that the more elements a set has, the closer the chosen mean to the mathematical expectation of the random value.
## Example 1
Height mean from the dataset:
```python
mean_height = df['height'].sum() / df['height'].shape[0]
mean_height
# can also be done
df['height'].mean()
```
Let's see how the selected mean changes when changing the size of the set:
```python
mean_list = []
for i in range(1, 62):
mean_list.append(df['height'][0:i*100].mean())
plt.plot(list(range(100, 6200, 100)), mean_list)
plt.show()
```
![[Pasted image 20260126203103.png]]
**Selective dispersion** evaluates the dispersion of a random value:
$$
\sigma_{X}^2=\frac{1}{n}\sum_{i=1}^n (x_{i}-\overline X)^2
$$
## Example 2
Selective dispersion of hockey players height:
```python
((df['height'] - df['height'].mean()) ** 2).sum() / df['height'].shape[0]
# or else
df['height'].var()
```
The values are different because such evaluation is biased. So, in practical tasks use **unbiased dispersion evaluation** is used:
$$
\sigma_{X, unbiased}^2=\frac{1}{n-1}\sum_{i=1}^n(x_{i}-\overline X)^2
$$
## Example 3
Unbiased evaluation of height:
`((df['height'] - df['height'].mean()) ** 2).sum() / (df['height'].shape[0] - 1)`
Difference between biased and unbiased evaluations:
Every object from the set - a random value. So, every statistics is a random value.
Evaluation is called unbiased if the mathematical expectation is equal to the real value of the parameter.
For example, let set X be derived from value x. Then selective mean is unbiased evaluation of mathematical expectation:
$$
M(\overline X)=M(x)
$$
This means that if we choose a large amount of selections then selective mean is unlikely to be equal to mathematical expectation of x.
Ordinary dispersion evaluation is biased:
$$
M(\sigma_{X}^2)=\frac{n-1}{n}D(x)
$$
*Note*: When evaluating dispersion unbiased metrics are used. So, we will consider $\sigma_{X}^2$ as unbiased evaluation:
$$
\sigma_{X}^2=\frac{1}{n-1}\sum_{i=1}^n(x_{i}-\overline X)^2
$$
We can control displacement and evaluations alike using `ddof` arg (Delta Degrees Of Freedom). In this parameter we indicate how much do we need to deduct from elements of the selection. For example, we can get a biased evaluation using `ddof=0`:
`df['height'].var(ddof=0)`
In general, dispersion is not a very visual measure of spread as it has a different dimension. So **standard deviation*** is commonly used with or instead of dispersion. It is equal to the square root of the initial parameter.
Biased and Unbiased evaluations:
$$
\sigma_{X}=\sqrt{ \frac{1}{n}\sum_{i=1}^n (x_{i}-\overline X)^2}, \:\:\:\ \sigma_{X, unbiased}=\sqrt{ \frac{1}{n-1}\sum_{i=1}^n (x_{i}-\overline X)^2}
$$
## Example 4
Standard deviation of height: (unbiased)
`np.sqrt(((df['height'] - df['height'].mean()) ** 2).sum() / (df['height'].shape[0] - 1))`
or 
`df['height'].std(ddof=1)`
# Mode, median, quantile
**Mode** - most frequently found value in a set
## Example 5
For starters, let's see how frequent are certain values of height using `.value_counts` method:
`df['height'].value_counts().head(10)`
or `df['height'].mode()`
Mode allows to get information about a set "average"
`df['height'].mean()` is roughly the same value

**Median** - a value t, that half of the elements from the set are less or equal t, the other half is greater or equal.
Median is the middle of a set: if we sort set elements median will be in the middle.
## Example 6
Let's find the median of the height. We sort the elements and take the middle of the array. Find the set size:
```python
height = sorted(df['height'])

length = len(height)
length
```
Number of elements is even, so median will be between 2 halves of the set.
`height[length // 2 - 1 : length // 2 + 1]` # `[184, 184]`
This means that the mean is 184. Let's see how many percent is on the left and on the right of the median:
```python
median = 184

(df['height'] <= median).sum() / length
(df['height'] >= median).sum() / length
df['height'].median()
```
The median is also a medium measure, as is the selective mean and mode.
If the amount of elements is odd, median is a value in the middle of a sorted set. If the amount of the elements is even, any value between limit right value of the left half and the left value of the right side.

Median is a special case of a more general concept - *quantile*
Median is a quantile of order 0.5. 
Let $\alpha \in (0,1)$. **Quantile of order $\alpha$** - a number $t_{\alpha}$ so that "$\alpha$ percent" of all elements of a set is less than $t_{\alpha}$ and, consequently, "($1-\alpha$ percent)" is more than $t_{\alpha}$.
Quantile can also be a certain element of a set or be between them.
Frequently used:
- **first quartile** - quantile of order 0.25
- **second quartile** - same as median
- **third quartile** - quantile of order 0.75
Sometimes can be used:
- **decile** - same as quartiles, but we divide into 10 parts, not 4. For example, median is a 5-th decile
- **percentile** - just an another way to define quantile. Instead of a range we use a percent.
## Example 7
Evaluate 1 and 3 quartiles of height.
First quartile:
```python
length // 2
(length // 2 + 1) // 2
height[(length // 2 + 1) // 2 - 1 : (length // 2 + 1) // 2 + 2]
q1 = 180

(df['height'] <= q1).sum() / length
(df['height'] >= q1).sum() / length

```
Third quartile:
```python
height[(length * 3 // 2 + 1) // 2 - 1 : (length * 3 // 2 + 1) // 2 + 2]
q3 = 188

(df['height'] <= q3).sum() / length
(df['height'] >= q3).sum() / length
df['height'].quantile(0.75)
```
Can also use:
`df['height'].quantile([0.25,0.5,0.75])`
or `df['height'].describe()`
**Interquartile range** - range between 1 and 3 quartiles, 50% of set values are in there.
It is used to measure spread of values around the mean. Sometimes its use is more acceptable than the standard deviation as it doesn't include spikes in data.
```python
list_ = [1] * 1000 + [10000]
np.mean(list_), np.std(list_, ddof=1)
```
Interquartile range:
`np.quantile(list_, [0.25, 0.75])` `#array([1.,1.])`
### Quantile of a random value
**Quantile of order $\alpha$ of a random value $X$** is a value $t_{\alpha}$ so that:
$$
P(X\leq t_{\alpha})=\alpha, \:\:\: P(X\geq t_{\alpha})=1-\alpha
$$
Using quantiles may help to "reverse" the spread function.
Let spread for X value be:
$$
F_{X}(x)=P(X\leq x)
$$
We are given a random value X and a limit t. We need to get the probability that X value does not exceed t. So we need a spread function.
Quite frequently we need to solve an inverse problem: we have a random value X and a probability $\alpha \in (0,1)$, we need to get a limit t so that $P(X\leq t)=\alpha$. This is a quantile of order alpha
## Visual representation
Histograms are frequently used when visualizing data spread.
1. All various values are marked on the x-axis.
2. The whole axis is split into a given number of even segments.
3. For each segment we get a number of set values which are in this segment and this number is represented on the y-axis.
### Example 8
Histogram of height: using `.hist`, as an argument `bins` we give a number of even segments, to which the x-axis is divided.
```python
from matplotlib import pyplot as plt
plt.style.use('seaborn-whitegrid')
df['height'].hist()
```
![[Pasted image 20260130175846.png]]
Using `bins`=20 to get a more detailed picture:
![[Pasted image 20260130175927.png]]
A histogram resembles a graph of spread of a random value, we just need to norm the y values to get the sum of all the columns to be 1, this can be done using `density`
```python
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))

df['height'].hist(ax=ax1)

df['height'].hist(ax=ax2, density=True)
```
![[Pasted image 20260130181404.png]]
Another way to visualize data - **boxplot**. Quartiles are marked in the box. "Whiskers" are the limits of the segments:
$$
[Q_{1}-1.5 \times IQR, \:\: Q_{3}+1.5 \times IQR],
$$
Where IQR - interquartile distance.
### Example 9
Create a boxplot of height:
`df[['height']].boxplot`
![[Pasted image 20260130181808.png]]
or using seaborn:
`sns.boxplot(df['height'], orient='v', width=0.15)`
![[Pasted image 20260130181834.png]]
All values over the limits are outliers, let's count them:
```python
q1 = df['height'].quantile(0.25)
q3 = df['height'].quantile(0.75)

iqr = q3 - q1

boxplot_range = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
boxplot_range
outliers = df.loc[(df['height'] < boxplot_range[0]) | (df['height'] > boxplot_range[1])]

outliers.shape[0]
outliers

# percentile of outliers:
outliers.shape[0]/df.shape[0]
```
