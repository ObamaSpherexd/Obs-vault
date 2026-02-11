**CODE WITH OUTPUTS IS IN COLAB:** https://colab.research.google.com/drive/1zOaZcjLdEI0MPV-sJVSkRUFVfmGY2b1n?usp=sharing


`pandas.Series.str.split` - splits the string around the given separator
Series.str.split(_pat=None_, _*_, _n=-1_, _expand=False_, _regex=None_)  regex determines whether there are regular expressions
```python
s='Hello world word'
print(s.split('w'))
print(s.replace('wo','v'))
tmp=df['topics'].str.replace(', ','/') # replaces , with / in list of topics
tmp=tmp.str.split('/',expand=True) # splits by /, extends rows to needed max len
print(tmp.iloc[:,0].unique)

print(tmp.iloc[:,1].unique)

print(tmp.iloc[:,2].unique) # print unique values in each column after splitting
tmp2=df['topics'].str.split(r'[\w,][,][\s]+',regex=True,expand=True) # deletes last letter if it is s then splits
tmp3=df['topics'].str.get_dummies(',') # filters by criteria and puts a number of its calls
df['topics']=df['topics'].str.replace(', ',',')

tmp4=df['topics'].str.get_dummies(',') # same as last

```
