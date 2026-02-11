#ML 
# Regression
in cases of supervised learning in ML there is a certain amount of marked data, for which we have a concrete answer and correct prediction. we try to train our model on this data and then predict based on new never seen before data.

we'll look into one of ML tasks (specifically supervised learning) - regression task

**regression task** (prediction) - creation of a model able to predict a numerical value based on a set of features.
## Regression. Task development

we have a teaching set with objects presented as feature description (feature vector) and a target variable value (continuous variable)

regression algorithm for a new object predicts the value of a target variable
**Examples**:
- estimating house value: by region, ecological state
- predicting bond features: by features of chemical elements evaluate melting temp, conductivity
- medicine: healing time by procedures provided
- credit score: credit max by survey
- engineering: mpg by technical features and riding style

**answer in regression tasks - value of continuous range, unlike classification tasks, where the answer is discrete.**
# linear regression methods
## linear regression

**linear regression** - the most simple tool for discovering dependencies between values. frequently considered to be an ML method, it is not.

recover linear regression - linear dependence between different numerical values - can be done using an analytical approach.

linear regression solves the same problem as ML - recovering dependencies between data - same definitions.

### basic definitions in ML

main definition in ML is *training set*. these are examples based on which we plan to build main dependence. $X$ is made out from $l$ object $x_{i}$ pairs and known answers $y_{i}$
$$X = (x_{i}, y_{i})^l_{i=1}.$$
a function that maps the object space $\mathbb{X}$ to a response space $\mathbb{Y}$, enabling predictions, is called a *algorithm* or a *model* $a(x)$. it takes an object as input and outputs an answer.

note that $x_{i} = (x^{1}, x^{2}, ..., x^{d})$. so every object $x_{i}$ consist of a series of different values.

simple linear model represented as:
$$f(x) = w_{0} + w_{1}x_{1} + ... + w_{d}x_{d} = w_{0}+\sum^{d}_{i=1}w_{i}x_{i}.$$
weights $w_{i}$ are parameters of model $f(x)$ . weight $w_{0}$ is called *free coefficient*, *bias* . optimization consists of finding optimal weights values. The sum in the formula can be written as a scalar multiplication of a feature vector $x=(x_{1},...,x_{d})$ with a weight vector $w=(w_{1},...,w_{d})$:
$$f(x) = w_{0}+\left \langle w,x \right \rangle.$$
To make a model uniform and simplify optimization we include a fictional feature $x^{0}$ which is always = 1. So:
$$f(x) = \left \langle w,x \right \rangle = \sum^{d}_{i=0}w_{i}x^{i} $$
- $y \in \mathbb{R}$ — target variable, the value can be predicted
- $x^1, x^2, \dots, x^n$ — features, based on which the prediction is made (input data)
- $f(x^1, x^2, \dots, x^n)$ — function explaining the dependence between features and the target variable (model)
### error function
