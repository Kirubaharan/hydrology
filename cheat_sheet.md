#Cheat Sheet

##Frequently used functions

###Drop a particular column
```python
DF = DF.drop('column_name', 1)
```
###Aggregate Daily Data
```python
rain_df = rain_df.resample('D', how=np.sum)   # D for day
```
###Select column using column name
```python
rain_df = df_base[['Date_Time', 'Rain Collection (mm)']]
```
