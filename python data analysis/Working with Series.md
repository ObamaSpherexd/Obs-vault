Series structure behave like vectors and can be added, multiplied by a number etc
```python
s7=pd.Series([10,2,30,40,50,8],['a','b','c','d','e','a'])
print(s7)
s7=s7**2
print(s7)
''' List - slow and bad
Numpy array - fast and good
pandas array - just good
'''
l1=[1,2,3]
l2=[4,5,6]

s7=pd.Series([10,20,30,40,50],['a','a','c','d','e'])
s8=pd.Series([1,1,1,1,0],['a','b','c','h','d'])
print(s7+s8) # for proper addition their len should be =
# only elelents with the same tag are operated
l=[1,2,4]
print(l*3)
print(s7*3)
```
