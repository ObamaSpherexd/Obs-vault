#Statistics 
# Statistical hypothesis testing. P-values. Confidence intervals.
## Statistical hypothesis testing
**Statistical hypothesis** - assumption about the dispersion type and the characters of the random value, which can be confirmed or denied based on the given information.
For example:
1. hypothesis - mathematical expectation of a random value is 10.
2. hypothesis - random value has standard dispersion.
3. hypothesis - 2 random values have an equal mathematical expectation.
The hypothesis being checked will be called **zero** $H_{0}$
## Example 1
We have a CNC which creates balls for bearings, which is fitted to make balls with a diameter of 1 mm. based on the set of diameters of the balls, we need to check whether the machine is calibrated correctly.
In this case the zero hypothesis $H_{0}$ is a hypothesis that the mathematical expectation of the diameter is 1 mm.
in parallel, we also observe the opposing $H_{1}$ hypothesis, which is called **alternative** or **rival**
in our case the alternative hypothesis states that the mathematical expectation of the diameter of the ball is not 1 mm.
based on the task, alternative hypothesis can be **left-sided**, **right-sided** or **double-sided**.
In our case the alternative hypothesis is double-sided as the mathematical expectation be either smaller or bigger than 1 mm
when checking a hypothesis we need to determine whether the deviation is negligible or not.
if the deviation is non-negligible then $H_{0}$ is denied in favour of the alternative and vice versa.
**Hypothesis check stages**:
1. zero and alternative hypothesis are formed.
2. a statistic $S(X)$ is given, which in case the zero hypothesis is true has a known spread (we know its spread function $F_{s}(x)=P(S<x)$)
3. fixed **significance level $\alpha$** - permissible for this task probability of the type 1 error. 
4. determine critical area $\Omega_{\alpha}$ so that $P(S \in \Omega_{\alpha}|H_{0})=\alpha$
5. a **statistical test** is conducted: for a set X value S(X) is calculated and if it is in $\Omega_{\alpha}$ then we discard $H_{0}$ and accept $H_{1}$
## Example 1
```python
import numpy as np
samples = np.array([0.6603, 0.9466, 0.5968, 1.3792, 1.5481, 0.7515, 1.0681, 1.1134,1.2088, 1.701 , 1.0282, 1.3579, 1.0191, 1.1784, 1.1168, 1.1372,0.7273, 1.3958, 0.8665, 1.5112, 1.161 , 1.0232, 1.0865, 1.02  ])

samples.mean()
samples.std()
```
### Selecting statistic S
Selection of a statistic depends on what kind of hypothesis is being checked, what spread we have and what info about the set we have.
For example, if we check a hypothesis around mathematical expectation of normally distributed random value with a known dispersion, then we use **Z-statistic**:
$$
Z=\frac{\overline X- {\mu}}{{\sigma /}\sqrt{ n }}
$$
where X - selection, $\overline X$ - selected mean, $\mu$ - mathematical expectation dictated by $H_{0}$, $\sigma$ - known std, n - amount of elements in a selection.
if the dispersion is unknown, we use **t-statistic**:
$$
t=\frac{\overline X-\mu}{\sigma_{X}/\sqrt{ n }}
$$
where $\sigma_{X}$ is unbiased evaluation of quadratic deviation.
if the $H_{0}$ hypothesis is correct then t-statistic has **Student's distribution** or **t-distribution** with parameter df=n-1
If we know dispersion, then we know mean square deviation and vice versa.
## Example 1
let standard quadratic deviation be 0.25, so then:
$$
Z=\frac{\overline X- {1}}{{0.25 /}\sqrt{ n }}
$$
```python
def statistic(samples: np.ndarray) -> float:
    return (samples.mean() - 1) / (0.25 / np.sqrt(samples.shape[0]))
```
### Selecting importance level
type 1 and 2 errors appear in tasks where we need to check whether an event happened or not.
**Type 1 error (false positive)** - a situation is set as "happened" when in reality it did not
**Type 2 error (false negative)** - opposite situation
in case of a metal detector:
1. type 1 error - detector worked on a person without illegal items
2. type 2 error - detector didn't work on a person with illegal item
we need to balance these types of errors
when checking statistical hypothesis we consider an event a case where a zero hypothesis was rejected.
Importance level - probability of type 1 error.
let alpha be 0.05
### Critical area
Critical area of a random value X are areas where the random value usually doesn't appear.
for normal distribution we can use 2 sigma rule:
chance to get in the interval $(\mu - 2\sigma; \mu+2\sigma)$ is 0.95. so we can look into critical area $(-\infty,\mu-2\sigma)\cup(\mu+2\sigma,\infty)$ with a probability of 0.05
this area is double-sided.
Critical area can be of these types:
- left-sided: $\Omega_{\alpha}=(-\infty,t_{\alpha})$
- right-sided: $\Omega_{\alpha}=(t_{1-\alpha},\infty)$
- double-sided: $\Omega_{\alpha}=(-\infty,t_{\alpha/2})\cup(t_{1-\alpha/2},\infty)$
$t_{\beta}$ is a quantile of order $\beta$ ($F_{S}(t_{\beta})=\beta$)
## Example 1
$$
Z=\frac{\overline X- {1}}{{0.25 /}\sqrt{ n }}
$$
this statistic has standard distribution
```python
from scipy import stats
t1 = stats.norm.ppf(alpha / 2)
t2 = stats.norm.ppf(1 - alpha / 2)

t1, t2
```
so, critical area:
$$
\Omega_{\alpha}=(-\infty,-1.96)\cup(1.96,\infty)
$$

### Statistical test
we have:
1. fixed statistic S(X)
2. built critical area $\Omega_{\alpha}$
if value is in critical area - discard null hypothesis

## Example 2
in reality we rarely have anything other than the set, we don't know the dispersion so we can conduct a statistical test, but we'll need a different statistic:
$$t = \dfrac{\overline{X} - \mu}{\sigma_X / \sqrt{n}},$$
```python
def statistic(samples: np.ndarray) -> float:
    return (samples.mean() - 1) / (samples.std(ddof=1) / np.sqrt(samples.shape[0]))

n = samples.shape[0]

t1 = stats.t.ppf(alpha / 2, df=n - 1)
t2 = stats.t.ppf(1 - alpha / 2, df=n - 1)

t1, t2
```
now the critical area is much larger:
$$\Omega_\alpha = (-\infty, -2.07) \cup (2.07, \infty)$$
which means that the test is more sensitive
## P-values
p values allow to get the result of multiple statistical checks for many levels of importance.
the lower the value the broader the critical area. **p-value** is max value of $\alpha$, when the hypothesis can be accepted
## Example 3
```python
n = samples.shape[0]
S = statistic(samples)

n, S
print('alpha\ hypothesis result')
print('-------------')

for alpha in np.linspace(0, 0.15, 15):
    t1 = stats.t.ppf(alpha / 2, df=n - 1)
    t2 = stats.t.ppf(1 - alpha / 2, df=n - 1)
    
    print(round(alpha, 4), '\t', t1 <= S <= t2)
```
with small $\alpha$ we have a large critical area in which it is hard to land. for the first few steps where the critical area is wide, the value of the statistic doesn't land there. at some point the critical area is so small that it devours the value of the statistic and the hypothesis is rejected.
P-value will be the $\alpha$ value at which this shift happens
let $F_{S}(x)$ - spread function, $t_{\beta}$ - quantile of order $\beta$. to get p-value:
1. for right-sided $\Omega_\alpha = \left( t_{1 - \alpha}, \infty \right)$ we have a case $t_{1 - \alpha} = S$, so $$P_r = 1 - F_S(S)$$
  2. for left sided $\Omega_\alpha = \left( -\infty, t_\alpha \right)$, condition $t_\alpha = S$, so $$P_l = F_S(S)$$

  

3. for double-sided $\Omega_\alpha = \left( -\infty, t_{\alpha / 2} \right) \cup \left( t_{1 - \alpha / 2} , \infty \right)$ we need a combination: $$P = 2 \cdot \min (P_l, P_r)$$
in our case the area is double-sided, so:
```python
p_left = stats.t.cdf(S, df=n - 1)
p_right = 1 - stats.t.cdf(S, df=n - 1)

pvalue = 2 * min(p_left, p_right)

pvalue
```
In practice: if the selected importance level is lower than the p-value gotten from the test, we can accept the hypothesis, otherwise it should be rejected.
## Confidence intervals
all mentioned above are **point** estimates, which means that a parameter was evaluated by a singular value.
Downside - can't understand how accurate is the evaluation.
to fix this problem we use confidence intervals. **Confidence interval** - an interval which with a set probability has a value of the parameter in focus.
let p (**confidence level**) is given. a confidence interval for $\theta$ parameter is a pair of statistics L and U such so:
$$
P(L\leq \theta\leq U)=p
$$
let set X is given from a normally distributed random value with a known dispersion $\sigma^2$ and we need to build a confidence interval for mathematical expectation $\mu$ with a confidence level p. then the statistic
$$Z = \dfrac{\overline{X} - \mu}{\sigma / \sqrt{n}}$$
has standard normal distribution.
let $\alpha=1-p$, we can check that:
$$P \left( t_{\alpha / 2} \leq Z \leq t_{1 - \alpha / 2} \right) = p,$$
where $t_{\beta}$ is a quantile of order $\beta$ for standard normal distribution. then:
$$P \left( t_{\alpha / 2} \leq \dfrac{\overline{X} - \mu}{\sigma / \sqrt{n}} \leq t_{1 - \alpha / 2} \right) = p$$

$$P \left( t_{\alpha / 2} \cdot \dfrac{\sigma}{\sqrt{n}} \leq \overline{X} - \mu \leq t_{1 - \alpha / 2} \cdot \dfrac{\sigma}{\sqrt{n}} \right) = p$$

  

$$P\left( \overline{X} - t_{\alpha/2, \, n-1} \cdot \frac{s}{\sqrt{n}} \leq \mu \leq \overline{X} + t_{\alpha/2, \, n-1} \cdot \frac{s}{\sqrt{n}} \right) = 1 - \alpha$$
In case of an unknown dispersion we use
$$t = \dfrac{\overline{X} - \mu}{\sigma_X / \sqrt{n}},$$
where $\sigma_{X}$ is mean quadratic deviation. This statistic has Student's distribution, so
$$P \left( t_{\alpha / 2, \: n - 1} \leq t \leq t_{1 - \alpha / 2, \: n - 1} \right) = p,$$
where $t_{\beta,n-1}$ is quantile of order $\beta$ with a parameter df=n-1. Confidence interval can be found similarly:
$$P \left( \overline{X} + t_{\alpha / 2, \: n - 1} \cdot \dfrac{\sigma_X}{\sqrt{n}} \leq \mu \leq \overline{X} + t_{1 - \alpha / 2, \: n - 1} \cdot \dfrac{\sigma_X}{\sqrt{n}} \right) = p$$
## Example 4
We are given a set of 10 random values of X.
if we think that random value X is normally distributed, we can create a confidence interval for M(X) with a confidence level of 0.95 using t-distribution.
let's find mean and unbiased evaluation for mean quadratic deviation:
```python
samples = np.array([6.9, 6.1, 6.2, 6.8, 7.5, 6.3, 6.4, 6.9, 6.7, 6.1])
n = samples.shape[0]

mean = samples.mean()
std = samples.std(ddof=1)

n, mean, std
# find quintiles using scipy:
p = 0.95
alpha = 1 - p

t1 = stats.t.ppf(alpha / 2, df=n - 1)
t2 = stats.t.ppf(1 - alpha / 2, df=n - 1)

t1, t2
# so, confidence interval is:
(mean + t1 * std / np.sqrt(n), mean + t2 * std / np.sqrt(n))
```
**confidence intervals for dispersion** are made using chi-square distribution.