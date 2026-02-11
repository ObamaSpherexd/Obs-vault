
```python
re.match()
re.search()
re.findall()
re.split()
re.sub()
re.compile()
```
### Search function
This function searches for the first `pattern` in `string` with additional `flags`:
`re.search(pattern,string,flags=0)`
Returns `match` object if there is any, else `None`
`match` conducts search in the beginning of the string, `search` - in the whole string
`search()` searches in the whole string, but returns only the first fit
```python
re.search(r'FULL', 'FULL HD 1920x1080 FULL') # <re.Match object; span=(0, 4), match='FULL'>
re.search(r'(\d+)\D(\d+)', 'IPS Panel Full HD / Touchscreen 1920x1080') # <re.Match object; span=(32, 41), match='1920x1080'>
```

### Findall function
Returns all found matches, no constraints.
`re.findall(pattern,string,flags=0)`
```python
re.findall(r'[0-9]*[x][0-9]*', '1920x1080 IPS Panel Touchscreen / 4K Ultra HD 3840x2160')
# ['1920x1080', '3840x2160']

def get_formatted_screen(text):
	retult=re.findall(r'[0-9]*[x][0-9]*',text)
	return result[0]
df['ScreenResolution'].map(get_formatted_screen)
```

### Split function
Splits the string according to the pattern
`re.split(pattern, string, [maxsplit=0])`
```python
re.split(r'[\s]', '1920x1080 IPS Panel Touchscreen / 4K Ultra HD 3840x2160') # split by space

```

### Sub function
Searches for a pattern and replaces it with an inputted value, if the pattern is not found then no changes.
`re.sub(pattern,repl,string)`
```python
re.sub(r'GB','','16GB')
def get_formatted_ram(value):
	value=re.sub(r'GB','',value)
	return value
df['Ram']-df['Ram'].map(get_formatted_ram)
```
