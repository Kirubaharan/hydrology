#Cheat Sheet

##Frequently used functions

###Drop a particular column
```python
DF = DF.drop('column_name', 1)
df.drop([Column Name or list],inplace=True,axis=1)   #will delete one or more columns inplace.
```
Dont use `inplace` if you dont want to change the original df

###Aggregate Daily Data
```python
rain_df = rain_df.resample('D', how=np.sum)   # D for day
```
###Select column using column name
```python
rain_df = df_base[['Date_Time', 'Rain Collection (mm)']]
```
###Change column name using column index no
```python
df_base.columns.values[x] = 'Air Temperature (C)'
```
x = column no

#### Print column name/ access column name by column index no
```python
print df_base.column.values[x]
```
###Merge dataframes pandas lib
```python
rain_weather = weather_daily_df.join(rain_days, how='right')
```
|left |		Use keys from left frame only|
|right|	 	Use keys from right frame only|
|outer|		Use union of keys from both frames|
|inner|	 	Use intersection of keys from both frames|