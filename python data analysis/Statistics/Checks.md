#Statistics 
# Links between values. correlation markers. correlation analysis. normality check
multidimensional observations are common in statistics, thus multidimensional statistical analysis is frequently used when we need to:
1. study dependence between causes and how they affect a variable
2. classify object with multiple features
3. lower dimensions of features if there are too many
Some topics were already discussed in [[Elements of probability theory and mathematical statistics]]
## Correlation
**Correlation** - (already mentioned in [[Elements of probability theory and mathematical statistics|here]]), mathematical parameter by which we can determine whether there is linear correlation between 2+ random values. its coefficient lies between `[-1,1]`
if correlation coefficient is near 1 then there is linear correlation between values, if it is near -1, there is opposing correlative link: one value bigger, other smaller. if it is near 0, there is no linear correlation.
## Example 1
height correlation
```python
df = pd.read_csv('drive/MyDrive/hockey_players.csv', encoding="ISO-8859-1")
df.head()

df[['height', 'weight']].corr()
'''we get a correlation matrix, which shows that the bigger the height the bigger the mass
'''

# including age:
df[['height', 'weight', 'age']].corr()
df['age'].min()
```
correlation with age is low.
## Links between values
if 2 values correlate this can mean that there can be statistical link between them. but this is true only for one particular set and may not be true for other sets.
High correlation between values can not be interpreted as a causal relationship between them.
for example, there is high correlation between the material losses in house fires and the amount of firefighters involved. false conclusion - more men more damage.
high correlation can mean the same cause when there is no direct contact between values.
winter is the cause for flu cases and rising heat prices. this 2 values have high correlation but don't influence each other.
lack of correlation doesn't mean that there is no link between values, maybe there is non linear dependence.
## Example 2
correlation between BMI and height:
$$
BMI=\frac{weight}{height^2}
$$
```python
df[['height', 'weight', 'bmi']].corr()
import numpy as np
np.corrcoef(df['height'] ** 2, df['bmi'])
```
correlation coefficient doesn't catch the dependence between height and BMI as the dependence is quadratic, not linear.
## Correlation markers
**covariance** - measure of linear dependence between random values.
covariance between X and Y:
$$
cov(X,Y)=M((X-M(X))(Y-M(Y)))
$$
evaluation can be biased or unbiased. unbiased:
$$
\sigma_{XY}=\frac{1}{n-1}\sum_{i=1}^n(x_{i}-\overline X)\times(y_{i}-\overline Y)
$$
where X,Y - sets of size n
## Example 3
covariance between height and weight
```python
X = df['height']
Y = df['weight']

MX = X.mean()
MY = Y.mean()

cov = ((X - MX) * (Y - MY)).sum() / (X.shape[0] - 1)
cov
# or:
import numpy as np
np.cov(X, Y, ddof=1)
'''this function returns covariation matrix. on the diagonal - dispersions, outside - paired covariances'''
X.var(), Y.var()
```
covariance is heavily dependent on the spread level of each value. so it is better to use **Pearson's correlation coefficient**:
$$
r_{XY}=\frac{\sigma_{XY}}{\sigma_{X}\cdot \sigma_{Y}}
$$
$\sigma_{X}, \sigma_{Y}$ - mean quadratic deviation
## Example 4
we get correlation coefficient for previous sets. 
```python
corr = cov / (X.std() * Y.std())
corr
'''Pearsons coef is evaluated using `.corr` '''
df[['height', 'weight']].corr()
'''or with `numpy.corrcoef` '''.
np.corrcoef(X, Y)
```
pros:
- uses a lot of info
- allows to test on correlation importance: statistic
  $$
  t=\frac{r\sqrt{ n-2 }}{\sqrt{ 1-r^2 }}
  $$
  has Student's distribution with n-2 degrees of freedom
cons:
- **sets have to have normal distribution**
- measures level of linear dependence
## Rank correlation
other than linear dependence there is also **rank** (or **orderly**) dependence. one value up other up, but the degree of increase doesn't have to be linear.
## Example 5
create a set of exponential spread of size 100. second set is from the first^ 5
```python
from matplotlib import pyplot as plt
plt.style.use('seaborn-whitegrid')
x = np.random.exponential(size=100)
y = x ** 5

plt.scatter(x, y)
# Pearsons coef:
np.corrcoef(x, y)
```
![[Pasted image 20260204002402.png]]
Pearson's is small as it represents the level of the linear dependence.
popular rank correlation coefs: Kendall and Spearman. Let $(x_{1},y_{1}),\dots ,(x_{m},y_{m})$ - pairs of values of 2 sets. two pairs are called concordant if $x_{i}<x_{j},y_{i}<y_{j}$ or the other way. otherwise they are called non-concordant.
let P - number of all concordant combinations of 2 pairs, Q - number of all non-concordant pairs. **Kendall's correlation coefficient**:
$$
\tau=\frac{P-Q}{P+Q}
$$
*note*: works only if there are no dupes.

calculate kendall's coef:
```python
def is_concordant(pair1: tuple, pair2: tuple) -> bool:
    """are pairs concordant.
    """
    
    
    return (pair1[0] - pair2[0]) * (pair1[1] - pair2[1]) > 0
'''amount of concordant and non-concordant pairs is found using `combinations` from `itertools` '''
from itertools import combinations
list(combinations(range(6), r=2))
list(zip(x, y))[:10]
P = 0
Q = 0

for pair1, pair2 in combinations(zip(x, y), r=2):
    if is_concordant(pair1, pair2):
        P += 1
    else:
        Q += 1
        
P, Q
# coef:
tau = (P - Q) / (P + Q)
tau
'''done realisation is presented in pandas'''.
pd.DataFrame({'x': x, 'y': y}).corr(method='kendall')
```
pros:
- doesn't need normal distribution
- rank dependence is a general case of linear
cons:
- uses less information compared to Pearson's coef
- straight tests on correlation value are not very real
## Normality test
we've mentioned a couple times that usability of a particular method is heavily dependant on whether the spread is normal.
normality tests are divided into 3 classes:
1. graphical methods::
   - Histogram
   - Q-Q curve
2. methods based on spread rules (standard deviation, 2 sigma, 3 sigma)
3. statistical methods:
   - colmogorov-smirnov
   - shapiro-whilk
**graphical methods** use graphs and diagrams to conduct normality tests.
## Example 6
create histograms of some features of the dataset. on top of them we overlay functions of density for normal distribution with the adequate parameters.
```python
from scipy import stats
keys = ['age', 'weight', 'height']

fig, axes = plt.subplots(ncols=len(keys))
fig.set_size_inches(4 * len(keys), 4)
axes = axes.flatten()

for key, ax in zip(keys, axes):
    ax.hist(df[key], density=True)
    
    loc = df[key].mean()
    scale = df[key].std()
    
    x_left, x_right = ax.get_xlim()
    x = np.linspace(x_left, x_right, 10000)
    y = stats.norm.pdf(x, loc=loc, scale=scale)
    
    ax.plot(x, y, linestyle='dashed')
    ax.set_title(key)
```
![[Pasted image 20260204005809.png]]
Other way - **Q-Q curve** (or quantile-quantile curve):
1. we evaluate mean alpha and mean quadratic deviation sigma for a set
2. for each $\alpha \in(0,1)$ we put on x-axis a quantile of order alpha for normal distribution with parameters $a,\sigma$, on y-axis - selected quantile of order alpha
the resulting set should lay on the line $f(x)=x$. we can determine the proximity of dots to the given line using paired regression.
## Example 7
build a Q-Q curve for a previous set.
```python
fig, axes = plt.subplots(ncols=len(keys))
fig.set_size_inches(4 * len(keys), 4)
axes = axes.flatten()

for key, ax in zip(keys, axes):
    samples = df[key]
    
    loc = samples.mean()
    scale = samples.std()
    
    interval = np.linspace(0, 1, samples.shape[0])[1:-1]
    x = stats.norm.ppf(interval, loc=loc, scale=scale)
    y = np.quantile(samples, interval)
    
    ax.scatter(x, y, s=5)
    ax.plot(x, x, color='C1', linestyle='dashed')
    
    ax.set_title(key)
    ax.set_xlabel('теоретические квантили')
    ax.set_ylabel('квантили выборки')
```
![[Pasted image 20260204010517.png]]
another way to test normality is use knows rules for spread:
- $P((\mu-\sigma),(\mu+\sigma)=0.68$
- $P((\mu-2\sigma),(\mu+2\sigma))=0.95$
- $P((\mu-3\sigma),(\mu+3\sigma))=0.997$
## Example 8
```python
for key in keys:
    print(key)
    
    samples = df[key]
    
    loc = samples.mean()
    scale = samples.std()

    for i in range(1, 4):
        true_value = stats.norm.cdf(i) - stats.norm.cdf(-i)
        sample_value = ((samples >= loc - i * scale) & (samples <= loc + i * scale)).sum() / samples.shape[0]
        
        print(f'{i} sigma(s)')
        print(f'\ttheoretical:\t{true_value}')
        print(f'\tsample:\t\t{sample_value}')
        
    print()
```
