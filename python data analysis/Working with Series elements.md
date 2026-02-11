[[Series data structure|Series]] elements can be called by index like a regular list

```python
l=[1,5,6,7]
print(l[0],l[-1],l[1:3])
s5=pd.Series(['Ian','Pete','Marie','Nastia','Nate'], ['a','b','c','d','a'])
print(s5)

print(s5[2])
print(s5['c'])
print(s5[-3])
print('---------')
print(s5['a'])
```

We can use tags so that working with **Series** will be similar to working with dicts
```python
s5=pd.Series(['Иван', 'Петр', 'Мария', 'Анастасия', 'Федор', 'Надя'], [1, 2, 4, 2, 6, 6])
print(s5)
print(s5[6])
# print(s5[-1]) will not work
```

*We can access Series elements using [ ] by either:*
*1. Giving an index*
*2. Giving a named tag if there are any*

#### Accessing elements by given conditions

We can put a condition in brackets, the result will be all elements fitting the criteria.
Any logical operation returns `True/False` and then the program only takes indexes with `True` value
```python 
s6=pd.Series([0,-13,1,-5,0],['a','b','c','d','e'])
print(s6[s6==0])
print(s6==0) # mask
l=[10,0,13,1,5,0]
for i in l:
	if i==0:
		print(i)
print('------------')
print(s6[['a','c','e']]) # returns elements with given tags

print('----------------------------------')
s6=pd.Series([10,13,1,5,0,6,4,8],['a','b','c','b','c','f','g','h'])
print(s6)
# type(s6[0]) is list
print(s6['a'::2])

```

