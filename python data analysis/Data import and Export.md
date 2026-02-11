We can use .csv or .xlsx files to create DataFrames
**CSV** (Comma-Separated Values) - text format for presenting data in table.
Each row is distinct from another, columns are separated by special simbols like a comma.
To read .csv files we use `read_csv()`
* filepath_or_buffer - path to file 
* sep - used separator
![[09.ods]]
```python 
df=pd.read_excel('09.ods',header=None)
print(df.head())
print(df.shape()) # (16000,6)
df2=dp.read_csv('File.csv')

df2.max() # =99 maximum value in a cell
df2.sum() # sum of all values in a row
df2.max(axis=0) # finds max in every column, be default axis=1
df2.mean(axis=0) # mean in every column
df.duplicated() # checks if a value if duplicated False/True
# find dublicates in all columns
duplicateRows=df[df.duplicated()]
duplicateRows=df[df.duplicated(['col1','col2'])] # finds dupes in particular columns
print(df.columns) # prints names of columns
```
## Data export
DataFrame and Series can be saved as .csv and .xlsx files
`DataFrame.to_csv(path_or_buf=None, sep=',', na_rep='', float_format=None, columns=None, header=True, index=True, index_label=None, mode='w', encoding=None, compression='infer', quoting=None, quotechar='"', line_terminator=None, chunksize=None, date_format=None, doublequote=True, escapechar=None, decimal='.', errors='strict', storage_options=None)`
* path_or_buf - path to file
* sep - line with len1, divider for outcoming file
To write a singular object to Excel .xlsx, we need to inly give the end file name. To write in multiple lists we need to create an ExcelWriter object with the name of the end file and point out the list in output file.
`DataFrame.to_excel(excel_writer, sheet_name='Sheet1', na_rep='', float_format=None, columns=None, header=True, index=True, index_label=None, startrow=0, startcol=0, engine=None, merge_cells=True, encoding=None, inf_rep='inf', verbose=True, freeze_panes=None, storage_options=None)`