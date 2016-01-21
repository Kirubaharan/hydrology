__author__ = 'kiruba'
import pandas as pd
from scipy.spatial import distance
import utm

data_file = '/home/kiruba/Documents/R/distance_matrix_gw_wells/Wells_list_Hadonahalli.csv'
data_df = pd.read_csv(data_file, sep=',')
coords = zip(data_df['Lat'], data_df['Long'])
coords_utm = [utm.from_latlon(lat, long) for lat, long in coords]
coords_utm_new = [x[0:2] for x in coords_utm]
print coords_utm_new
distance_matrix = distance.cdist(coords_utm_new, coords_utm_new, 'euclidean')
# print data_df.head()
# print coords
distance_matrix_df = pd.DataFrame(distance_matrix, columns=data_df['Wp_No'], index=data_df['Wp_No'])
distance_matrix_df.to_csv('/home/kiruba/Documents/R/distance_matrix_gw_wells/python_data_matrix.csv')


