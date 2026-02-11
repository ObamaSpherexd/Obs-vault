#ML 
gradient descent - one of the most popular algorithms used in ML. when training a model, we use a variation of gradient descent.
we'll look into simple examples and more complex description of the method to understand it.
# Simplified example of searching maximum through calculating derivative of 1d function
**gradient descent** for 1 variable is a method of optimizing a function.
**optimization** in math, CS and operational science - task of finding an extremum of target function in a particular area of a finite-dimensional vector space, limited by a set of linear/non-linear equations.
## Theory
we need to have movement of the X-axis towards the minimum. we can use derivatives to do so. the sign of the first derivative helps us understand where is the minimum.
if we need to move to the minimum and we are to its left, then we need to move to the right (+) and vice versa.
$x_{n+1}=x_{n}-f'(x_{n})$
big numbers harm the movement (see below) and to avoid this problem we include a correction factor $\lambda$
$x_{new}=x_{n}-\lambda*f'(x_{n})$
this coeff is called training speed.

---
in math - a gradient is a vector, which shows the direction and magnitude of the fastest growth of a scalar function in a particular point, made out of partial derivatives.
___
## Practical implementation
functions:
$f(x)=x^2-6*x+5$
$\frac{\partial f}{\partial x}=f'(x)=2*x-6$
$f_{2}(x=x^2+15*\sin(x)$
$f_{2}'(x)=2*x+15*\cos(x)$
```python
import numpy as np
import matplotlib.pyplot as plt
# functions for calculating values of the initial functions and their derivarives.

def f(x):
  return x*x - 6*x + 5

def dfdx(x):
  return 2*x - 6

def f2(x):
  return x*x + 15*np.sin(x)

def df2dx(x):
  return 2*x + 15*np.cos(x)
```
## Practical implementation of gradient descent
```python
# key parameters

N = 200 # iterations
lr = 0.6 # optimization speed
import numpy as np

x_plot = np.arange(-5, 11, 0.01)
x_plot

y_plot = [f(x) for x in x_plot]
y_plot[:2]

fig, ax = plt.subplots()
ax.set_title(f'Скорость движения: {lr}, количество итераций: {N}')

x_plot = np.arange(-5, 11, 0.01)
y_plot = [f(x) for x in x_plot]
ax.plot(x_plot, y_plot)
```
![[Pasted image 20260207184728.png]]
```python
fig, ax = plt.subplots()
ax.set_title(f'speed: {lr}, iterations: {N}')

x_plot = np.arange(-5, 11, 0.01)
y_plot = [f(x) for x in x_plot]
ax.plot(x_plot, y_plot)

xc = -4 #начальное значение
print(f'starting min: {f(xc)} in point {xc}')

ax.scatter(xc, f(xc), c='r')

for _ in range(N):
  x0 = xc # graph

  xc = xc - lr*dfdx(xc) # descent

  ax.scatter(xc, f(xc), c='b') # graph
  ax.plot([x0,xc],[f(x0), f(xc)], c='r') # graph

ax.scatter(xc, f(xc), c='g')

print(f'final minimum: {f(xc)} in point {xc}')

plt.show()
```
![[Pasted image 20260209185447.png]]
## Creating a function for calculations
```python
fig, ax = plt.subplots()
ax.set_title(f'Скорость движения: {lr}, количество итераций: {N}')

x_plot = np.arange(-5, 11, 0.01)
y_plot = [f(x) for x in x_plot]
ax.plot(x_plot, y_plot)

xc = -4 #начальное значение
print(f'Начальное значение минимумума: {f(xc)} в точке {xc}')

ax.scatter(xc, f(xc), c='r')

for _ in range(N):
  x0 = xc #нужно только для графика

  xc = xc - lr*dfdx(xc) #сам по себе градиентный спуск

  ax.scatter(xc, f(xc), c='b') #нужно только для графика
  ax.plot([x0,xc],[f(x0), f(xc)], c='r') #нужно только для графика

ax.scatter(xc, f(xc), c='g')

print(f'Финальное значение минимумума: {f(xc)} в точке {xc}')

plt.show()
#### Создание функции для расчёта
def gd(N, lr, min = -5, max = 11, f=f, dfdx=dfdx, xc=-4):
    """
    realisation and visuals for gradient descent 

    Parameters
    ----------
    N: integer
        steps needed

    lr: float
        learning rate

    min: float
        min value on graph

    max: float
        max value on graph

    f: function
        target func

    dfdx: function
        targnt func gradient

    Returns
    ----------
    """
    fig, ax = plt.subplots()
    ax.set_title(f'speed: {lr}, iterations: {N}')

    x_plot = np.arange(min, max, 0.01)
    y_plot = [f(x) for x in x_plot]
    ax.plot(x_plot, y_plot)

    print(f'starting min: {f(xc)} in point {xc}')

    ax.scatter(xc, f(xc), c='r')

    for _ in range(N):
      x0 = xc # graph

      xc = xc - lr*dfdx(xc) # descent

      ax.scatter(xc, f(xc), c='b') # graph
      ax.plot([x0,xc],[f(x0), f(xc)], c='r') # graph

    ax.scatter(xc, f(xc), c='g')

    print(f'final min: {f(xc)} in point {xc}')
    plt.show()
gd(20, 1, min=-5, max = 11)

# we can also analyse how the rate affects the result of optimisation

for i in range(1, 10):
  gd(20, i/10)
```
![[Pasted image 20260209190052.png]]![[Pasted image 20260209190056.png]]
![[Pasted image 20260209190102.png]]
![[Pasted image 20260209190107.png]]![[Pasted image 20260209190111.png]]
![[Pasted image 20260209190119.png]]
![[Pasted image 20260209190124.png]]
## Difficulties of optimization (local minimums)
a lot depends on the starting point when working with functions with multiple minimums:
```python
gd(20, 0.05, min=-15, max= 15, f=f2, dfdx=df2dx, xc=10)
```
![[Pasted image 20260209190725.png]]
```python
gd(20, 0.03, min=-15, max= 15, f=f2, dfdx=df2dx, xc=-5)
```
![[Pasted image 20260209190747.png]]
# Gradient descent in Machine Learning
let's look into a more formal task of finding target function using gradient descent.

let's introduce target function $h_{\theta}(x)$. this function depends on parameters $\theta$. to find dependencies we can use a data set $(x^i ,y^i )$. our task is to find optimal values of $\theta$ when $h_{\theta}(x)$ is closest to $y^i$.

${x^{(i)}}$ is called a feature vector. ${y^{(i)}}$ is called a target function values vector

to solve the task we need to introduce "loss function", which will depict the difference between real data {$y^{(i)}$} and calculated values {$h_{\theta}(x^{i})$} or {$h(\theta)^i$}.

for this function to be usable in gradient descent it has to be continuous and differentiable (has a derivative in every point). a common function is square loss function, it is calculated for the whole set.

**Loss function:** (including a correction factor, derived from [[Elements of probability theory and mathematical statistics|basics of mathematical statistics]]) $$J(\theta) = \frac {1}{2m} \sum_{i=1}^{m} (h(\theta)^i - y^i)^2$$

**Gradient descent** - a way to minimize a loss function $J(\theta)$, which is dependent of $\theta$, by updating parameters in a "direction", opposite to the gradient of the target function $\nabla_{\theta}J(\theta)$. learning rate $\eta$ defines the steps size made to achieve min.

**Metaphor** for the process: "We follow along the slope created by the target function until we reach a valley".

Gradient is a vector made out of partial derivatives of a function. for example:
$$ f(x, y) = x^2+y^3 $$ gradient will look like: $$ (\frac{\partial{f(x, y)}}{\partial{x}} , \frac{\partial{f(x, y)}}{\partial{y}}) = (2x, 3y^2)  $$

__Loss function gradient:__ $$\frac{\partial{J(\theta)}}{\partial{\theta_i}} = \frac {1}{m} \sum_{i=1}^{m} (h(\theta^i) - y^i) X_j^i$$

If the dependence is linear and can be be written as an equation:

$$h_\theta(x) = \theta_0 + \theta_1X$$

, then the parameters will change as the following on each step:

$$\theta_0 = \theta_0 - \eta (\frac {1}{m}\sum_{i=1}^{m}((h(\theta^i) - y^i)X_0^i))$$

$$\theta_1 = \theta_1 - \eta (\frac {1}{m}\sum_{i=1}^{m}((h(\theta^i) - y^i)X_1^i))$$

starting values $(\theta_0, \theta_1)$ are usually random. the procedure stops when the values stop changing.
```python
from copy import deepcopy # best module to copy objects
from tqdm import tqdm # library for creation of a progress bar

import seaborn
import numpy as np
import matplotlib.pyplot as plt
std_error = 30 # coeff to create random values
sample_size = 10000 # set size
theta0, theta1 = 1, 10 # func params
x = np.random.randn(sample_size) # generating random set
func_y = lambda x: theta0 + theta1 * x # lambda function to create func values
# def func_y(x):
#     return theta0 + theta1 * x
y = func_y(x) + std_error * np.random.randn(sample_size) # calculating values including random spread
# visualisation

x_plot = np.linspace(x.min(), x.max(), 1000)
y_plot = func_y(x_plot)

fig = plt.figure(figsize=(10, 7))
plt.scatter(x, y, alpha=0.25)
plt.plot(x_plot, y_plot, "r--", linewidth=2, label=f"{theta0} + {theta1}x")
plt.xlim(x.min(), x.max())
plt.xlabel("x", size=14)
plt.ylabel("y", size=14)
plt.legend(loc="best")

plt.show()
```
![[Pasted image 20260209194524.png]]
```python
def calculate_predictions(theta0, theta1, X):
    """
    calculating func values for coefs and values of X.

    Parameters
    ----------
    theta0: float

    theta1: float

    X: array-like, shape = [n_samples, ]
        features vector.

    Returns
    -------
    y_pred: array-like, shape = [n_samples, ]
        func value.

    """
    return theta0 + theta1*X


def calculate_cost(theta0, theta1, X, y):
    """
    calculating loss func value.

    Parameters
    ----------
    theta0: float

    theta1: float

    X: array-like, shape = [n_samples, n_features]
        feature vector.

    y: array-like, shape = [n_samples, ]
        target vector.

    Returns
    -------
    cost: float
        loss func value.

    """
    theta0 = np.atleast_3d(np.asarray(theta0)) # transform into suitable for usage 
    theta1 = np.atleast_3d(np.asarray(theta1))

    y_pred = calculate_predictions(theta0, theta1, X)
    cost = np.average((y - y_pred)**2, axis=2)/2

    return cost


def gradient_descent_step(theta, X, y, learning_rate):
    """
    a step of gradient descent.

    Parameters
    ----------
    theta: array-like
        array of theta.

    X: array-like, shape = [n_samples, n_features]
        feature vector.

    y: array-like, shape = [n_samples, ]
        target vector.

    learning_rate: float
        learning rate.

    Returns
    -------
    updated_theta: array-like
        updated array of theta.

    """
    n = len(y)
    y_pred = calculate_predictions(theta[0], theta[1], X)

    updated_theta = deepcopy(theta)
    updated_theta[0] -= learning_rate / n * np.sum((y_pred - y))
    updated_theta[1] -= learning_rate / n * np.sum((y_pred - y) * X)

    return updated_theta
```

```python
def plot_gradient_descent(cost_history, theta_history, X, y):
    """
    visualizing process and result of descent
    4 graphs:
    1. loss func on every iteration
    2. theta on iteration
    3. result changing of the accumulating line
    4. level lines and gradient descent process

    Parameters
    ----------
    cost_history: list[float]
        list with loss func values on every interation.

    theta_history: list[np.array]
        list of 2d arrays of theta.

    X: array-like, shape = [n_samples, n_features]
        feature vector.

    y: array-like, shape = [n_samples, ]
        target vector.

    """
    fig = plt.figure(figsize=(15, 15))
    plt.subplot(221)
    plt.scatter(range(len(cost_history)), cost_history)
    plt.xlabel("iterations", size=14)
    plt.ylabel(r"J($\theta$)", size=14)

    plt.subplot(223)
    plt.scatter(x, y, alpha=0.15)
    x_plot = np.linspace(-3, 3, 500)
    for num, theta in enumerate(theta_history):
        y_plot = calculate_predictions(theta[0], theta[1], x_plot)
        if num == 0:
            plt.plot(x_plot, y_plot, color="green", label="start line - baseline", linewidth=5)
        if num == len(theta_history) - 1:
            plt.plot(x_plot, y_plot, color="orange", label="Result", linewidth=5)
        else:
            plt.plot(x_plot, y_plot, "r--", alpha=.5)
    plt.xlim(x_plot.min(), x_plot.max())
    plt.xlabel("x", size=14)
    plt.ylabel("y", size=14)
    plt.plot(x_plot, func_y(x_plot), color = "black", label="initial line", linewidth=3)
    plt.legend(loc="best")

    plt.subplot(222)
    x_plot = range(len(cost_history)+1)
    plt.scatter(x_plot, [theta[0] for theta in theta_history], label=r'$\theta_0$')
    plt.scatter(x_plot, [theta[1] for theta in theta_history], label=r'$\theta_1$')
    plt.xlabel("iterations", size=14)
    plt.ylabel(r"$\theta$", size=14)
    plt.legend(loc="best")
    plt.subplot(224)
    plot_cost_function(X, y, theta_history)


def plot_cost_function(X, y, theta_history):
    """
    visualizing gradient descent process and building level lines

    Parameters
    ----------
    X: array-like, shape = [n_samples, n_features]
        features vector.

    y: array-like, shape = [n_samples, ]
        target vector.

    theta_history: list[np.array]
        list of 2d arrays of theta.

    """
    theta0 = [theta[0] for theta in theta_history]
    theta1 = [theta[1] for theta in theta_history]

    #theta0_grid = np.linspace(-5*min(theta0), max(theta0), 200)
    #theta1_grid = np.linspace(-5*min(theta1), max(theta1), 200)
    theta0_grid = np.linspace(-25, 25, 200)
    theta1_grid = np.linspace(-25, 25, 200)
    cost_grid = calculate_cost(
        theta0_grid[np.newaxis,:,np.newaxis],
        theta1_grid[:,np.newaxis,np.newaxis],
        X, y
    )
    X, Y = np.meshgrid(theta0_grid, theta1_grid)

    theta0, theta1 = theta_history[-1]
    plt.scatter([theta0]*2, [theta1]*2, s=[50, 0], color=['k','w'])
    contours = plt.contour(X, Y, cost_grid, 30)
    plt.clabel(contours)

    for it in range(1, len(theta_history)):
        plt.annotate(
            '', xy=theta_history[it], xytext=theta_history[it-1],
            arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 2},
            va='center', ha='center'
        )

    plt.scatter(*zip(*theta_history), color="black", s=40, lw=0)
    plt.xlim(theta0_grid.min(), theta0_grid.max())
    plt.ylim(theta1_grid.min(), theta1_grid.max())
    plt.xlabel(r'$\theta_0$', size=15)
    plt.ylabel(r'$\theta_1$', size=15)
    plt.title('loss function', size=15)
```
```python
n_iterations, learning_rate = 100, 0.5
theta_history, cost_history = [100*np.random.rand(2)], []

for it in tqdm(range(n_iterations)):
    last_theta = theta_history[-1]
    current_theta = gradient_descent_step(
        theta=last_theta, X=x, y=y, learning_rate=learning_rate
    )
    theta_history.append(current_theta)
    cost_history.append(calculate_cost(current_theta[0], current_theta[1], X=x, y=y))
plot_gradient_descent(
    cost_history=cost_history,
    theta_history=theta_history,
    X=x, y=y
)
```
![[Pasted image 20260209200422.png]]
