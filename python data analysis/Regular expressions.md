**Regular expressions** - blueprints which are used to compare symbols in strings

Repos of most common regular expressions:
- [https://regex101.com](https://www.google.com/url?q=https%3A%2F%2Fregex101.com)
- [http://www.regexlib.com](https://www.google.com/url?q=http%3A%2F%2Fwww.regexlib.com)
- [https://www.regular-expressions.info](https://www.google.com/url?q=https%3A%2F%2Fwww.regular-expressions.info)
```python
import re
df=pd.read_csv('data_regex.csv')
```
Regular expressions usually contain special symbols which are called *metasymbols*: `[] {} () \ * + ^ $ ? . |`
Every predetermined symbol class starts from `\`, each class is the same with a symbol from a set
- `\b` - any number `0-9`
- `\D` - anything but a number
- `\s` - any space-symbol (tab, space, \n)
- `\S` - anything but a space
- `\w` - any word symbol (letter, number, _)
- `\W` - anything but a word symbol
```python
# check that screen resolution has 4 numbers
if re.fullmatch(r'\d{4}','1920'):
	print('yes')
else:
	print('no')

if re.fullmatch(r'\d{4}','768 '):
	print('yes')
else:
	print('no')
	
```
*Symbol class* - sequence in a regex, coinciding with 1 symbol. In order for the coincidence to be composed of several symbols, we need to specify a quantificator after the class
Quantificator {4} repeats `\d` 4 times 
`[]` define users symbol class, coinciding with one symbol. So `[aeiou]` = lower register vowel. `[A-Z]` - upper, `[a-z]` - lower, `[a-zA-Z]` - any letter
Let's check the name - the sequence starts with Upper register, followed by any amount of lower register:
```python
if re.fullmatch('[A-Z]*','Full HD 1920x1080'):
	print('yes')
else:
	print('no')

if re.fullmatch('[A-Z]*','FULLHD'):
	print('yes')
else:
	print('no')
```
Metasymbols in users symbol class are interpreted as literal symbols, so `[*+$]` means `+` `*` or `$`
Quantificators `*` and `+` are maximums - they coincide with the max possible amount of symbols.
`{n,}` coincides with at least n values
```python
if re.fullmatch(r'\d{3,}','Full HD 1920x1080':
	print('yes')
else:
	print('no')
```

### [[Functions of the `re` module]]

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
