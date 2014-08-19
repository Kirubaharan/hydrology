#Pandas Cheat Sheet

##Frequently used functions
`df` refers to Pandas data frame
###Drop a particular column
```python
df = df.drop('column_name', 1)
df.drop([Column Name or list],inplace=True,axis=1)   #will delete one or more columns inplace.
```
Dont use `inplace` if you don't want to change the original df

###Aggregate Daily Data
```python
df = df.resample('D', how=np.sum)   # D for day
```
###Select column using column name
```python
df1 = df[['column_name_1', 'column_name_2']]
```
###Change column name using column index no
```python
df.columns.values[x] = 'new column name'
```
x = column no

### Select column pandas
#### Access column name by column no(position starting from 0)
```python
print df.column.values[x]
```
#### Select a particular column for a given datetime index
```python
print df['2014-05-20']['column_name']
```

###Merge two dataframes
```python
df3 = df1.join(df2, how='right')
```
 Merge| Description
 ----- |-----
|left |		Use keys from left frame only|
|right|	 	Use keys from right frame only|
|outer|		Use union of keys from both frames|
|inner|	 	Use intersection of keys from both frames|