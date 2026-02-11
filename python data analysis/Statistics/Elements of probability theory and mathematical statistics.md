#Statistics
**A random event** - an event, which can either happen or not in certain conditions.
Examples:
- During a die roll 1 and 2 were rolled on 2 dice
- Bank client didn't return a loan
- A coin was flipped 100 times, heads 55 times
An event can be called **certain** if during an experiment it will surely happen
An **impossible** event will never happen
Examples of a **certain** event:
- A number less than 7 was rolled on a die
- During a coin flip either heads or tails were rolled
- You have scores less than 0 on an exam
Examples of an impossible event:
- 2 dice were rolled once, sum is 15
- A coin was flipped 100 times, 55 heads, 56 tails
- You have scored more than 101 on an exam
## Key parameters
### Mathematical expectation
A probability-weighted value of a random quantity
$$
E[X] = Σ_i{p_i*x_i}
$$
$p_{i}$ - probability of an i event occurring, $x_i$ - random event of the i-th experiment
### Dispersion
Random value dispersion - quantity of span of a random value relative to its mathematical expectation
$$
D[X]=E[(X-E[X])^2]
$$
Dispersion cannot be applied to real world values as real world values usually include standard deviation, meaning a square of dispersion divided by number of experiments 
### Mean and median
Mean - value in a magnitude of experiments, which occurs most frequently. Median - value, which in a sorted set is directly in the middle.

### Median or mathematical expectation?
Evaluating mean as a minimum is useless without understanding border values
## Correlation
A common task is to evaluate a degree of statistical coherence between 2 values. The most common approach in these tasks is correlation 'analysis'. 
Pirson's correlation is evaluated accordingly: 
$$
r_{xy} = \frac{Σ(X-E[X])(Y-E[Y])} {\sqrt{Σ(X-E[X])^2Σ(Y-E[Y])^2}} 
$$
$Σ(X-E[X])(Y-E[Y])$ is called covariance
Correlation - quantity of linear link between 2 random values
```python
import matplotlib.pyplot as plt

d = np.array([9, 4, 1, 0, 1, 4, 9])

e = np.array([-3, -2, -1, 0, 1, 2, 3])

x = np.array([-3, -2, -1, 0, 1, 2, 3])

plt.plot(x, d, x, e)

plt.show()

print(np.corrcoef(d,e))
'''
[[1. 0.]
 [0. 1.]]
'''
```
How can we interpret the output?
These are all the possible variants of correlation evaluation: $[[r_{xx},r_{xy}] / [r_{yx},r_{yy}]]$
We can see that correlation between d and e is nonexistent. However, it is obvious that d=e * e
## Normal distribution
**Normal distribution** - also called Gaussian distribution - is a probability distribution, which in a 1D-plane case is given by a function of a probability density, similar to Gauss function
Formula of normal probability distribution: $$f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

Normal distribution has an enormous value in mathematical statistics as a sum of a large amount of random vaguely dependent on each other and same size values will be normally distributed.
**Not everything in the world is normally distributed**
# Practice
## Thesis
2 students - one is always on time, the other one is not. We need to understand whether there is significant statistical difference between them. 
## Solution - type 1 error
We need to check 2 values on khi-square criteria. 
Khi-criteria can be found in **statsmodels** library. **stats.proportion** object help to evaluate proportions. 
A classical A/B test - use **proportions_chisquare** of **proportion** class. First arg - int or array which responds for "successes", second one - total experiments
**proportions_chisquare** will return 3 variables, one of them being pval.
**p-value** allows us to understand if there is significant difference between 2 students. By celecting alpha as 0.05 we allow 5% of our observations to be incorrect.
**p-value** is a probability of coming across type 1 error by changing null hypothesis. Null hypothesis in our case is a saying: diffference between 2 students is statistically significant (not accident)
- If p-value is less than the limit - result of the experiment can be counted as statistically significant.
- if p-value is greater - the diffrerce between 2 students is random
However, there is also **type 2 error**, which also has to be considered
**Power** is a chance to see change where it is present. In our situation this value will point out whether the difference between 2 students can be spotted during the selected period of time.
Standard limit - 0.8 (80% probability for the change to be noticeable)
Power check can be done by importing **stats.power** object from **statsmodels**.
![[Pasted image 20260125100238.png]]
![[Pasted image 20260125100206.png]]
We also need to use **solve_power()** method, which receives arguments - **nobs** (number of observations), **alpha**, **effect_size**
**effect_size** - difference between relative values of attendance of 2 students. It is evaluated as follows:
$$
ES=\sqrt{ \frac{(p_{0}-p_{1})^2}{p_{0}}}
$$
Using this formula we can find power
![[Pasted image 20260125101347.png]]
Such small value says that the chance to differentiate between 2 students is only 5%. So our experiment suffices the type 1 error and doesn't fit the criteria for type 2 errors.
We need to either change the number of observations, or the attendance
![[Pasted image 20260125101636.png]]
So, if the second student is late 41 days in a year, we can say that the difference is statistically significant and can be seen