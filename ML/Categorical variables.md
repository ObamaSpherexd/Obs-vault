#ML 
categorical variables (factor/nominal variables) - variables with discrete values, categories or classes.
examples:
- sex
- color
- education
while working with most of ML models it is necessary to transform categorical variables in numerical form, as many ML algorithms work with numerical presentation of data
```python
!pip install opendatasets --quiet
import opendatasets as od

dataset_url = 'https://www.kaggle.com/datasets/allexanderspb/studentsperformance'
od.download(dataset_url)

# {"username":"adele1997","key":"3d752ef4484f4ee0c4a1164c37d74f04"}
import pandas as pd

df = pd.read_csv('/content/studentsperformance/StudentsPerformance.csv')
df
df['parental level of education'].value_counts()
```
we have a lot of data:
- gender
- race/ethnicity
- parental level of education
- lunch
- test preparation score
- math score
- reading score
- writing score
we have no blanks in dataset, math reading writing scores are int.
other variables are categorical, so let's look into ways to code these variables into ints.
# Categorical features processing
the simplest way:
- manually create a dict of unique words
- each word = number
- change words into numbers
`df['gender'].unique`
we see that there are only 2 unique values.
creating a dict:
```python
# only 2 unique values
# creatind dict
d = {'female': 0, 'male': 1}

# change categorical values into numbers
df['gender'] = df['gender'].map(d)
# df['gender'] = df['gender'].str.replace(d)

df.head()
```
cons: when we have a lot of categories manual approach is very time-consuming 
## Encoding with ready-made methods
	1. **label encoding**
simple and one of the most popular approaches, where categorical data is transformed into numerical format. unique category - unique number
in `sklearn` library there is a module `preprocessing` with a premade algorithm `LabelEncoder()`, which is responsible for encoding variables. 
`from sklearn import preprocessing`
one of key features of the library is a unified approach for processing algorithms. most of the methods are presented as classes, which makes it usage intuitively understandable and linear:
1. creating a class object
2. fitting
3. using(predict for ML algorithms, transform for preprocessing data)
`df['test preparation course'].unique()`
```python
from sklearn import preprocessing
# from sklearn.preprocessing import LabelEncoder

label_encoder = preprocessing.LabelEncoder()


label_encoder.fit(df['test preparation course'])

# fit - обучение
```
on fitting stage the algorithm searches all different categories in a variable (like unique()), then sorts them in lexicographical order and arranges a number for each category.
using our model to transform the variable:
```python
df['test preparation course'] = label_encoder.transform(df['test preparation course'])

# inverse_transform
# df['test preparation course'] = label_encoder.inverse_transform(df['test preparation course'])
label_encoder.classes_
label_encoder
df.head()
```
`LabelEncoder()` is a smart encoder, it saves the dictionary of categories and numbers, it is also able to reverse the transformation
**cons**:
1. LabelEncoder() transforms categories into ints, so the algorithms looks at them as sorted values which is incorrect for categories without order. this is critical for methods dependent on distance.
2. categories `['red','green','blue']` are transformed into`[0,1,2]`. the algorithm may think that green is closer to red than to blue.
3. hard to work with multiple categories. so objects can respond to multiple categories (film in multiple genres)

4 .  One-hot encoding
method accepts every categorical variable as an argument, then creates a new variable for every category, so every fictional variable has either 1 or 0 - there is a category or there is no
![[Pasted image 20260207151131.png]]
2 algorithms to realize:
1. `get_dummies()` - method of `Pandas`. simple function, not a encoding algorithm, therefore it is called dumb, does not remember encoding steps, dicts and cannot reverse the transformation
2. `OneHotEncoder` from `sklearn` from module `preprocessing`. similar to `get_dummies()`, but the encoder is smart. `get_dummies()` - method of a class, used once
example:
```python
df['race/ethnicity'].value_counts()
import pandas as pd

df = pd.get_dummies(df, columns=['race/ethnicity'])

df.head()
```
we see that we have 5 new variables (columns) `race/ethnicity_group A\B\C\D\E`, each has a prefix of an initial name of the column, in items there are 0 and 1
smart encoder for `parental level of education`
```python
from sklearn import preprocessing

# 1. initializing encoder
one_hot_encoder = preprocessing.OneHotEncoder()

# 2. fitting the algorithm
one_hot_encoder.fit(df['parental level of education'].values.reshape(-1, 1))

# [1,2,3,4]
# [[1],[2],[3],[4]] - reshape(-1, 1)

# 1. create
# 2.  (fit())
# 3.  (transform())
# 3. use the model of encoder
x_new = one_hot_encoder.transform(df['parental level of education'].values.reshape(-1, 1)).toarray()
x_new
x_new.shape
```
**`OneHotEncoder` returns a new object, which has to be added to the initial dataset**:
```python
# get new columns names
encoded_columns = one_hot_encoder.get_feature_names_out(['parental level of education'])

# transromt into DataFrame
encoded_df = pd.DataFrame(x_new, columns=encoded_columns)

# merging with the initial DataFrame
df = pd.concat([df, encoded_df], axis=1)

# deleting the initial column (optional)
df.drop(columns=['parental level of education'], inplace=True)
df.head()
```
pros of using smart encoders from `sklearn`:
1. they same information about categories when using `fit()` and uses the same logic as when using `transform()`. this ensures that new data will be processes as the training data. unlike `pd.get_dummies`, which only work once with the provided data and doesn't save any parameters. encoding new data requires repeating all the steps.
2. they can work with unknown categories using `handle_unknown`. this is useful when there is a new category in the dataset. example:
   - `handle_unknown='ignore'`: ignores unknown categories
   - `handle_unknown='infrequent_if_exist'`: merges rare categories into one group
3. `OneHotEncoder`: by default returns sparse matrixes (if `sparce_output=True`). this saves memory and speeds up calculations with many categories.
4. smart encoders allow working with multiple categorical columns simultaneously, they also allow to create categories clearly and have lots of useful methods, such as `get_future_names_out()` to get proper column names or `inverse_transform`- method of reverse transforming. `pd.get_dummies`: minimal settings, simple tasks only
## Working with multiple categories
we have looked into different methods of encoding variables, discussed pros and cons of different approaches. one of major problems: working with multiple categories for one object:
```python
import pandas as pd

df= pd.DataFrame({
    'game': [
        'The Witcher 3',
        'Minecraft',
        'Overwatch',
        'Dark Souls',
        'Fortnite',
        'Civilization VI',
        'Stardew Valley',
        'GTA V',
        'League of Legends',
        'Among Us'
    ],
    'genres': [
        'RPG,Action,Adventure',
        'Sandbox,Survival,Creative',
        'Shooter,Multiplayer,Action',
        'RPG,Action,Adventure',
        'Battle Royale,Shooter,Multiplayer',
        'Strategy,Simulation,Turn-Based',
        'Simulation,Farming,Indie',
        'Action,Adventure,Open World',
        'MOBA,Multiplayer,Strategy',
        'Social Deduction,Multiplayer'
    ]
})

df
```
we have created a small dataset with games, where each game can respond for multiple genres (multiple categories from variable `genres`)
if we'll encode the variable using `LabelEncoder()`, then as it does not understand multiple categories, then it will encode the set for the whole object, which is wrong, so we should use `One Hot Encoding`
We can use either the smart from `sklearn` or the dumb encoder from `Pandas`:
```python
df['genres'].str.get_dummies(sep=',')
# pd.get_dummies()
tmp = df['genres'].str.get_dummies(sep=',')

# adding prefix to new columns
tmp.columns = tmp.columns.str.split().str[0]
tmp = tmp.add_prefix('genres_').reset_index(drop=True)

tmp
```
**2 nuances with using this method**:
1. method doesn't change the initial dataset, creates a new DataFrame with encoded variables. if we want to add encoded variables to the initial DataFrame we need to use pd.concat
2. names of new variables: new column names are names according to the name of the category. do not have the prefix of the initial column. if we have an another variable which we plan to encode similarly and it has categories with the same name, this will create multiple columns with the same name, which can cause errors.
**adding prefix (`add_prefix('genres_')`) is important because**:
- removes conflict between names of columns when combining dummy-variables from different features.
- preserves understanding - we can clearly see to which feature is each column connected.
- makes debugging and model training easier.
```python
import pandas as pd

df = pd.DataFrame({
    'genres': ['Action,Comedy', 'Drama', 'Comedy', 'Action,Drama'],
    'year': [2010, 2012, 2014, 2016]
})
df['country'] = ['Action', 'USA', 'Comedy', 'USA']

df
```
example: 2 columns with same categories. without prefix:
```python
tmp = df['genres'].str.get_dummies(sep=',')
tmp_country = df['country'].str.get_dummies()
df = pd.concat([df, tmp, tmp_country], axis=1)
df
```
2 columns with a same name - **bad**!!!!
```python
# adding prefixes
tmp = df['genres'].str.get_dummies(sep=',').add_prefix('genres_')
tmp_country = df['country'].str.get_dummies().add_prefix('country_')

df = pd.concat([df, tmp, tmp_country], axis=1)
df
```

```python
# using smart encoder from sklearn

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# transforming string into list 
df['genres_split'] = df['genres'].str.split(',')
df
type(df.loc[0, 'genres_split'])
```

```python
# initializing MultiLabelBinarizer
mlb = MultiLabelBinarizer()
# using MultiLabelBinarizer for column 'genres'
genres_encoded = mlb.fit_transform(df['genres_split'])
genres_encoded
# creating new DataFrame with encoded genres
genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)

# merging with initial DataFrame
df = pd.concat([df, genres_df], axis=1)

df.head()
```
