Usually data scientists study the data before working with it.
**Exploratory Data Analysis, EDA** - preliminary research of a Dataset to understand its main characteristics, correlation between them, used to shrink the amount of methods, used to create a Machine Learning Model.

**EDA usually includes:**
- checking data dimensions (GB, PB, etc.)
- analyzing data types
- analyzing data on NaN values
- analyzing anomalous data
- discovering categories in data
- analyzing duplicates
- analyzing correlations between variables
- etc…

**EDA is usually done using these methods:**
- 1D-visualisation - provides statistics for each field in a set of unfiltered data
- 2D-visualisation - used to find correlation between each variable in a data set and a variable of interest
- N-D-visualisation - used to understand processes between different fields in a data set
- Shrinking dimensions - helps to understand data fields, in which the most error occurs between experiments, helps process less data volume
Using these methods a specialist can check assumptions and discovers templates, which help to understand the problem and choose a model, which also confirms that the data was created the way it was intended. 
## Matplotlib
Matplotlib - a base module of graph creating in Python.

**Seaborn** - library used to create statistics graphs in Python, based on [[Matplotlib]] and heavily relies on [[Pandas]] data structures.
```python
plt.plot([1,2,3,4,5],[1,4,9,16,26],marker='o')
plt.show()
```


```python
# same but using numpy
x=np.linspace(0,5,50)
y=x
plt.plot(x,y,'b')
plt.title('This is linear correlation y=x')
plt.xlabel('this is x')
plt.ylabel('this is y')
plt.grid()
plt.show()
```

1. `Figure`
   This is the canvas for graphs
   On each canvas we can put multiple graphs
2. `Axis`
   This is one graph, including its axes and the area in which the data are visualized/
   One `Figure` can have multiple `Axis`
   We can change `Axis` axes, tags, titles, etc.
3. `Subplots`
   This is a way to automatically create multiples `Axis` on one `Figure`
   Helps to create a mesh of graphics
   `matplotlib.pyplot.subplot( *args, **kwargs)` - function of adding an `Axis` to an existing `Figure`
![[Pasted image 20251226005350.png]]

```python
x = np.linspace(0, 5, 50)
y1 = x 
y2 = [i**3 for i in x] # y2[1] = x1^3 ... 

fig = plt.figure(figsize=(10, 5)) 
 
# plt.subplot(nrow, ncols, index) -iterations from 1 
# plt.subplots(nrow, ncols) from 0 
plt.subplot(1, 2, 1) 
plt.plot(x, y1) 
plt.ylabel("y1", fontsize=14) 
plt.grid(True) 
plt.subplot(1, 2, 2)
plt.plot(x, y2, c='r') 
plt.xlabel("x", fontsize=14) 
plt.ylabel("y2", fontsize=14) 
plt.show()
```
![[Pasted image 20251226005639.png]]

## Dot scatter
We use a different method when creating this graphics type.
```python
x = np.random.rand(1000) # x - dot coords - height
y = np.random.rand(1000) # y - dot coords - age 
fig, ax = plt.subplots() 
# plt.subplots(n_rows,n_columns) 
# fig, axes = plt.subplots(3,2) 
# ax.plot() 
ax.scatter(x, y, c = 'deeppink') # цвет точек 
# sns.scatterplot() 
# ax.set_xlabel() - plt.xlabel() 
# ax.set_title() 
ax.set_facecolor('yellow') # цвет области Axes 
fig.set_facecolor('green') 
plt.show()
```

```python
x = np.random.rand(1000)
y = np.random.rand(1000)
fig, axes = plt.subplots(2,2) 
axes[1,1].scatter(x, y, c = 'deeppink') # цвет точек 
axes[0,0].scatter(x, y, c = 'y') # цвет точек 
axes[0,0].set_facecolor('black') # цвет области Axes
axes[0,0].set_title("coords flat 0-0") 
axes[0,1].set_ylabel('this y') 
fig.set_facecolor('r') 
plt.show()
```

```python
x = np.random.rand(5000) # create 4 data varints with gamma-distribution

y1 = np.random.gamma(1, size = 5000) 
y2 = np.random.gamma(2, size = 5000) 
y3 = np.random.gamma(4, size = 5000) 
y4 = np.random.gamma(8, size = 5000) 
fig, ax = plt.subplots() # s - size
# colors {'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'}: 
ax.scatter(x, y1, c = 'g', s = 1) 

ax.scatter(x + 1, y2, c = [[0.255, 0.99, 0.71]], s = 5) # hex RGB: 
ax.scatter(x + 2, y3, c = '#FF6347', s = 10)  
ax.scatter(x + 3, y4, c = ['0.7'], s = 20) 
fig.set_figwidth(10) 
fig.set_figheight(20) 
plt.show()
```

## Histograms
```python
x = np.arange(1, 5) 
y = np.random.randint(1, 20, size = 4) 
fig, ax = plt.subplots()
ax.bar(x, y) # hist(x) 
# ax.set_facecolor('black') 
fig.set_facecolor('floralwhite') 
plt.show()
```

```python
fig, ax = plt.subplots() 
x = [1, 2, 1, 5, 1, 6, 2, 1] 
# value_counts() # 1 - 4 # 2 - 2 # 5 - 1 # 6 - 1 
ax.hist(x) 
# ax.set_facecolor('black') 
fig.set_facecolor('floralwhite') 
plt.show()
```
We can create multiple histograms on different parts of a figure using subplots, otherwise they start to overlap each other.
```python
x1 = np.arange(1, 8) 
y1 = np.random.randint(1, 20, size = 7) 
x2 = np.arange(1, 51) 
y2 = np.random.randint(1, 20, size = 50) 
fig, axes = plt.subplots(2, 1) 
# fig, axes = plt.subplots(2, 2) [r, c] 
axes[0].bar(x1, y1) 
axes[1].bar(x2, y2) 
axes[0].set_facecolor('yellow') 
axes[1].set_facecolor('red') 
fig.set_facecolor('green') 
plt.show()
```
Lifehack on how to fit 2 data arrays on a single histogram:
```python
fig, ax = plt.subplots() 
ax.bar(x-0.2, y1, width = 0.4) # shift x position slightly from the centre of the bar
ax.bar(x+0.2, y2, width = 0.4) 
plt.show()
```
Using seaborn:
```python
import matplotlib.pyplot as plt 
import seaborn as sns 
x = ['А', 'Б', 'В'] 
y = [10, 50, 30] # plt.bar() 
sns.barplot(x=x, y=y)
```

```python
'''DOES NOT WORK IN OBSIDIAN AS SEABORN CAN NOT BE IMPORTED'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv")

# sns.set(style="white")
# sns.set(style="whitegrid", color_codes=True)

fig, axes = plt.subplots(1, 2, figsize=(5, 5))

# sns.countplot(df, x=df['Survived'], palette='hls')
# sns.countplot(df['Survived'], palette='hls')

# countplot = hist()
# ax.scatter()
sns.countplot(x = df['Sex'], ax=axes[0], palette='Set2')
sns.countplot(x = df['Survived'], ax=axes[1], palette='mako')

fig.show()
```

## Other types of graphics
### Correlogram
This type of graph is used to visualize correlations between different data. We need a different Seaborn module.
`df.corr(numeric_only=True)`
![[Pasted image 20251229195801.png]]
**Correlation** - relation between different criteria in statistics. For example, when one parameter increases, another one also increases/decreases. Correlation is used to understand dependencies between variables.
**Correlation can be:**
- positive - one parameter rises, another does too  (from 0 to +1)
- negative - one parameter rises, another decreases (from -1 to 0)
- neutral - differences are not linked (0)
 ```python
plt.figure(figsize=(12,10)) 
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix, cmap='icefire', center=0, annot=True) 
# color pallete sns 
# Decorations 
plt.title('Correlogram of mtcars', fontsize=22) 
plt.xticks(fontsize=12) plt.yticks(fontsize=12) 
plt.show()
 ```
Uses a combinations on correlation table and a heatmap
### Paired dependencies
Visualises spread of each pair of data sets
```python
df=sns.load_datasen('iris')
df = sns.load_dataset('iris') 
# Plot 
plt.figure(figsize=(10,8)) 
sns.pairplot(df, hue='species') 
plt.show()
```
### Null values
```python
  
plt.figure(figsize=(15,7)) 
cmap = sns.cubehelix_palette(as_cmap=True, light=.9) 
sns.heatmap(df.isna().transpose(), cmap=cmap, cbar_kws={'label': 'Missing Data'})
```
![[Pasted image 20251229201806.png]]
```python
  
plt.figure(figsize=(20,40)) 
sns.displot( data=df.isna().melt(value_name="missing"), y="variable", hue="missing", multiple="fill" ) 
plt.show()
```
![[Pasted image 20251229201905.png]]

### BoxPlot
**Box and Whisker Plot / Box Plot** - a way to show data through quartiles
Straight lines coming from the box are called 'Whiskers' and are used to show degree of dispersion over the upper and lower quartiles. Can be placed both vertically and horizontally.
Box plots are usually used in descriptive statistics and allow to quickly analyze one or more data sets. It may seem less informative compared to prior mentioned methods, but in terms of saving space it is much better
Types of observations, which can be done with a box with whiskers:
- which are key values like 25%0 mean
- are there any outbursts and which are their values
- is the data symmetrical
- how tightly is data packed
- is the data shifted to a particular direction
![[Pasted image 20251229203527.png]]
- mean (Q2/50-th percentile): mean value of a data set
- 1 quartile (Q1/25-th percentile): mean value between minimal value and mean of a data set
- 3 quartile (Q3/75-th percentile): mean value between mean and maximum value
- IQR (Interquartile range): from 25 to 75 percentile. IQR shows how are mean values distributed
- "maximum": Q3+1.5* IQR
- "minimum": Q1-1.5* IQR
- outbursts: (green dots) In statistics an outburst is a point which is far from other observations
![[Pasted image 20251229204818.png]]
![[Pasted image 20251229204859.png]]
- Mean is less affected by outbursts as it is displayed in the center, not mean arithmetical
- Upper quartile - a grade, higher than which are only 25%
- Lower quartile - a grade, lower than which are 75%
- IQR - difference between 75 and 25 quartiles. It this diapason lays 50% of observations. if a diapason is small, it means that parts of the subarray think alike on a grade. If it is wide - there is no common answer
- Outbursts - non-typical observations. For example: 
	- **25% - 1.5 x IQR**
	- **75% + 1.5 x IQR**
Results of statistical questionnaires and boxplots are usually presented together. P-value helps to understand if the data we had collected is truly random. If P-value is <0.05, then differences between groups are not accidental
```python
plt.figure(figsize=(10,10))

df['sepal_length'].plot(kind='box')
```
We can evaluate quartiles manually using [[Pandas]]:
```python
Q1 = df['Age'].quantile(0.25) 
Q3 = df['Age'].quantile(0.75) 
IQR = Q3 - Q1 
print(Q1) 
print(IQR) 
print(Q3) 
df[(df['Age'] < Q1-1.5*IQR) | (df['Age']> Q3+1.5*IQR)]
```
### One variable distribution
**Distribution** - how the data is distributed. We need to evaluate them in order not to make any mistakes during further calculations.
**Random variable** - variable, which can accept any value from a particular set
**Probability distribution** - shows probability of all random values of a random variable. It is a theoretical distribution, which is made mathematically, has a mean and dispersion - analogies of mean and dispersion in empiric distribution.
**Dispersion in statistics** - quantity, which shows spread between results. If they are close to mean, the dispersion is low, otherwise it is high
![[Pasted image 20251229211033.png]]
```python
plt.figure(figsize=(5,5)) 
# plt.dist() 
sns.distplot(df['sepal_width'], bins=20)
```