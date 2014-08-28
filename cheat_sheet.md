#Pandas Cheat Sheet
Mostly collected from Stackflow answers.

##Frequently used functions
`df` refers to Pandas data frame
###Drop a particular column
[Ref:](http://stackoverflow.com/a/18145399/2632856)
```python
df = df.drop('column_name', 1)
df.drop([Column Name or list],inplace=True,axis=1)   #will delete one or more columns inplace.
```
Dont use `inplace` if you don't want to change the original df

###Select column using column name
####Select single column
```python
df['column name']
```
#### Select multiple columns as a dataframe
```python
df1 = df[['column_name_1', 'column_name_2']]
```
###Change column name using column index no
[Ref:](http://stackoverflow.com/a/11346337/2632856) See the comments
```python
df.columns.values[x] = 'new column name'
```
x = column no

### Select column pandas
#### Access column name by column no(position starting from 0)
```python
print df.column.values[x]
```
###Date Time Index
#### Create datetime column from two columns have date and time separately
```pyton
format = '%d/%m/%y %H:%M:%S'
#pd = pandas
df['Date_Time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format=format)
```
#### Set datetime index
```python
df.set_index(df['Date_Time'], inplace=True)
```
#### Select a particular column for a given datetime index
```python
print df['2014-05-20']['column_name']
```
#### Aggregate Daily Data
[Ref:](http://stackoverflow.com/questions/17001389/pandas-resample-documentation)
```python
df = df.resample('D', how=np.sum)   # D for day
```

###Merge two dataframes
[Ref:](http://pandas.pydata.org/pandas-docs/stable/merging.html#brief-primer-on-merge-methods-relational-algebra)
```python
df3 = df1.join(df2, how='right')
```
 Merge| Description
 ----- |-----
|left |		Use keys from left frame only|
|right|	 	Use keys from right frame only|
|outer|		Use union of keys from both frames|
|inner|	 	Use intersection of keys from both frames|

### Move a last column to front
[Ref:](http://stackoverflow.com/a/13148611/2632856)
```python
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]  # bring last column to first
df = df[cols]
```

### Give binary value for conditional statements
[Ref:](http://stackoverflow.com/a/18473330/2632856)
```python
df['new_column_name'] = (df['column'] == value).astype(int)
```

### Extract date, month, year from Datetime Index(yyyy-mm-dd)
[Ref:](http://stackoverflow.com/a/21954923/2632856)
```python
def datesep(df):
    """

    :param df: dataframe
    :param column_name: date column name
    :return: date array, month array, year array
    """

    date = pd.DatetimeIndex(df.index).day
    month = pd.DatetimeIndex(df.index).month
    year = pd.DatetimeIndex(df.index).year
    return date, month, year
```    
### Augmented Assignment
[Ref](http://legacy.python.org/dev/peps/pep-0203/)

```python
#Instead of using `x = x + y` use
x +=y
#<x> <operator>= y
```
#### Slicing dataframe
### Using .ix
```python
df.ix[row,column]
```
###Matplotlib Plots
####Using Tex, Latex in matplotlib plots
##### String formatting in Latex
```python
"""
{x : .4} x refers to position of string in .format(starts with 0)
 and .4 refers to one decimal approximation, 
 use 0.5 for two decimal places
"""
plt.text(x=-0.25, y=3000, fontsize=15, 
         s=r"\textbf{{$ y = {0:.4} x^2 + {1:.4} x + {2:.4} $}}".format(coeff_stage_area_cal[0],
                                                                       coeff_stage_area_cal[1],
                                                                       coeff_stage_area_cal[2]))
```