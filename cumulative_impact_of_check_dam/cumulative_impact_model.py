__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from contextlib import contextmanager
import simpy
import scipy as sp
import math
from datetime import datetime
import checkdam.checkdam as cd
import networkx as nx

"""
Check dam network
"""
# hadonahalli_network = nx.Graph()
# hadonahalli_network.add_nodes_from([463, 639, 625, 627, 633, 640, 641, 634, 626, 624, 'H1', 'H2', 'H3', 'H4', 'H5', 'H12', 'H45', 'H123', ])
# hadonahalli_network.add_edges_from([(463,640),(639, 641), (627, 634), (634, 624), (633,626)])
# hadonahalli_network.add_edges_from([(640, 'H1'), (641, 'H2'), (625, 'H3'), (624,'H4'), (626, 'H5')])
# hadonahalli_network.add_edges_from([('H1', 'H12'), ('H2', 'H12'), ('H4', 'H45'), ('H5', 'H45'), ('H12', 'H123'), ('H3', 'H123')])
# nx.draw(hadonahalli_network, with_labels=True)
# plt.show()
# raise SystemExit(0)

"""
Parameters
"""
INFLOW_463 = [0, 0, 10, 5, 0, 10]
INFLOW_640 = [0, 0, 8, 3, 0, 8]
EVAP_463 = [0, 0, 1, 1, 1, 1]
EVAP_640 = [0, 0, 1, 1, 1, 1]
TOTAL_VOL_463 = 8.5
TOTAL_VOL_640 = 5
INITIAL_VOL = 0
TOTAL_DAYS = len(INFLOW_463)
print TOTAL_DAYS


@contextmanager
def columns_in(df):
    """
    Source: https://gist.github.com/aflaxman/4121076#file-sdm_diaper_delivery-ipynb
    :param df: Pandas dataframe
    :return: column names in df as a tuple
    """
    col_names = df.columns
    col_list = [df[col] for col in col_names]
    try:
        yield tuple(col_list)
    finally:
        for i, col, in enumerate(col_names):
            df[col] = col_list[i]

#
# column_names = ['current_volume_463', 'current_volume_640', 'inflow_463', 'inflow_640', 'evap_463', 'evap_640',
#                 'overflow_463', 'overflow_640']
# state = pd.DataFrame(index=range(-7, TOTAL_DAYS), columns=column_names)
# state.ix[-7:0, ['current_volume_463', 'current_volume_640']] = INITIAL_VOL
# state.ix[-7:0, ['inflow_463', 'inflow_640', 'evap_463', 'evap_640']] = 0
# state.ix[0:TOTAL_DAYS - 1, ['inflow_463']] = INFLOW_463
# state.ix[0:TOTAL_DAYS - 1, ['inflow_640']] = INFLOW_640
# state.ix[0:TOTAL_DAYS - 1, ['evap_463']] = EVAP_463
# state.ix[0:TOTAL_DAYS - 1, ['evap_640']] = EVAP_640
#
# with columns_in(state) as (current_volume_463, current_volume_640, inflow_463, inflow_640, evap_463, evap_640, overflow_463, overflow_640):
#     for t in range(TOTAL_DAYS - 1):
#         current_volume_463[t + 1] = current_volume_463[t] + inflow_463[t] - evap_463[t]
#         current_volume_640[t + 1] = current_volume_640[t] + inflow_640[t] - evap_640[t]
#         if current_volume_463[t + 1] < 0:
#             current_volume_463[t + 1] = 0
#         if current_volume_640[t + 1] < 0:
#             current_volume_640[t + 1] = 0
#         if current_volume_463[t + 1] > TOTAL_VOL_463:
#             overflow_463[t + 1] = current_volume_463[t+1] - TOTAL_VOL_463
#             current_volume_463[t+1] = TOTAL_VOL_463
#         if current_volume_640[t + 1] > TOTAL_VOL_640:
#             overflow_640[t + 1] = current_volume_640[t+1] - TOTAL_VOL_640
#             current_volume_640[t+1] = TOTAL_VOL_640
#
# t = state.filter(['current_volume_463', 'current_volume_640', 'inflow_463', 'inflow_640', 'overflow_463', 'overflow_640'])
# t.plot(linewidth=2, alpha=0.9)
# plt.show()
date_format = '%Y-%m-%d %H:%M:%S'
# Weather file
weather_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/hadonahalli/corrected_weather_ws.csv'
# Rain file
rain_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/hadonahalli/ksndmc_rain.csv'
# convert to pandas dataframe
weather_df = pd.read_csv(weather_file, sep=',', header=0)
weather_df['Date_Time'] = pd.to_datetime(weather_df['Date_Time'], format=date_format)
weather_df.set_index(weather_df['Date_Time'], inplace=True)
weather_df.sort_index(inplace=True)
weather_df = weather_df.drop('Date_Time', 1)
# print weather_df.head()
# raise SystemExit(0)
# Rain data frame
rain_df = pd.read_csv(rain_file, sep=',', header=0)
# set index
rain_df['Date_Time'] = pd.to_datetime(rain_df['Date_Time'], format=date_format)
rain_df.set_index(rain_df['Date_Time'], inplace=True)
# sort based on index
rain_df.sort_index(inplace=True)
# drop date time column
rain_df = rain_df.drop('Date_Time', 1)
# print rain_df.head()
# raise SystemExit(0)
"""
Remove Duplicates
"""
weather_df['index'] = weather_df.index
weather_df.drop_duplicates(subset='index', take_last=True, inplace=True)
del weather_df['index']
weather_df = weather_df.sort()
print weather_df.head()

class Open_Water_Evaporation(object):
    def __init__(self,check_dam_name, air_temperature, relative_humidity,incoming_solar_radiation, wind_speed_mps, date_time_index, elevation, latitdude, longitude):
        self.check_dam_name = check_dam_name
        self.date_time_index = date_time_index
        self.elevation = elevation
        self.latitude = latitdude
        self.longitude = longitude
        self.air_temperature = air_temperature
        self.relative_humidity = relative_humidity
        self.incoming_solar_radiation = incoming_solar_radiation
        self.wind_speed_mps = wind_speed_mps
        self.air_p_pa = self.calculate_air_pressure()
        self.air_pressure = np.empty(len(self.date_time_index))
        self.air_pressure.fill(self.air_p_pa)
        self.extraterrestrial_irradiation = self.calculate_extraterrestrial_irradiation()
        self.half_hour_EO = self.calculate_half_hour_EO()

    def calculate_air_pressure(self, elevation=None):    # None is the key here
        z = elevation or self.elevation
        p = ((1 - (2.25577 * (10 ** -5) * z)))
        air_p_pa = 101325* (p ** 5.25588)
        return air_p_pa

    def calculate_extraterrestrial_irradiation(self, date_time=None, latitude=None, longitude=None):
        lat = latitude or self.latitude
        long = longitude or self.longitude
        date_time = date_time or self.date_time_index
        l = np.size(date_time)
        s = 0.0820  # MJ m-2 min-1
        lat_rad = lat * (math.pi / 180)
        if l < 2:
            day = (date_time - datetime(datetime.year, 1, 1)).days + 1
            hour = float(date_time.hour)
            minute = float(date_time.minute)
            b = ((2 * math.pi) * (day - 81)) / 364
            sc = 0.1645 * (math.sin(2 * b)) - 0.1255 * (math.cos(b)) - 0.025 * (math.sin(b))  # seasonal correction in hour
            lz = 270  # for India longitude of local time zone in degrees west of greenwich
            lm = (180 + (180 - long))  # longitude of measurement site
            t = (hour + (minute / 60)) - 0.25
            t1 = 0.5  # 0.5 for 30 minute 1 for hourly period
            w = (math.pi / 12) * ((t + (0.0667 * (lz - lm)) + sc) - 12)
            w1 = w - ((math.pi * t1) / 24)  # solar time angle at beginning of period [rad]
            w2 = w + ((math.pi * t1) / 24)  # solar time angle at end of period [rad]
            dr = 1 + (0.033 * math.cos((2 * math.pi * day) / 365))  # inverse relative distance Earth-Sun
            dt = 0.409 * math.sin(((2 * math.pi * day) / 365) - 1.39)  # solar declination in radian
            ws = math.acos(-math.tan(lat_rad) * math.tan(dt))
            if (w > ws) or (w < -ws):
                rext = 0.0
            else:
                rext = ((12 * 60) / math.pi) * s * dr * (((w2 - w1) * math.sin(lat_rad) * math.sin(dt)) + (math.cos(lat_rad) * math.cos(dt) * (math.sin(w2) - math.sin(w1))))  # MJm-2(30min)-1
        else:
            rext = np.zeros(l)
            for dt in date_time:
                i = date_time.get_loc(dt)
                day = (dt - datetime(dt.year, 1, 1)).days + 1
                hour = float(dt.hour)
                minute = float(dt.minute)
                b = ((2 * math.pi) * (day - 81)) / 364
                sc = 0.1645 * (math.sin(2 * b)) - 0.1255 * (math.cos(b)) - 0.025 * (math.sin(b))  # seasonal correction in hour
                lz = 270  # for India longitude of local time zone in degrees west of greenwich
                lm = (180 + (180 - long))  # longitude of measurement site
                t = (hour + (minute / 60)) - 0.25
                t1 = 0.5  # 0.5 for 30 minute 1 for hourly period
                w = (math.pi / 12) * ((t + (0.0667 * (lz - lm)) + sc) - 12)
                w1 = w - ((math.pi * t1) / 24)  # solar time angle at beginning of period [rad]
                w2 = w + ((math.pi * t1) / 24)  # solar time angle at end of period [rad]
                dr = 1 + (0.033 * math.cos((2 * math.pi * day) / 365))  # inverse relative distance Earth-Sun
                dt = 0.409 * math.sin(((2 * math.pi * day) / 365) - 1.39)  # solar declination in radian
                ws = math.acos(-math.tan(lat_rad) * math.tan(dt))
                if (w > ws) or (w < -ws):
                    rext[i] = 0.0
                else:
                    rext[i] = ((12 * 60) / math.pi) * s * dr * (((w2 - w1) * math.sin(lat_rad) * math.sin(dt)) + (math.cos(lat_rad) * math.cos(dt) * (math.sin(w2) - math.sin(w1))))  # MJm-2(30min)-1
        return rext

    def calculate_half_hour_EO(self, airtemp=None, rh=None, airpress=None, rs=None, rext=None,u=None, z=None):
        at = airtemp or self.air_temperature
        rh = rh or self.relative_humidity
        ap = airpress or self.air_pressure
        rs = rs or self.incoming_solar_radiation
        rext = rext or self.extraterrestrial_irradiation
        u = u or self.wind_speed_mps
        z = z or self.elevation
        half_hour_eo = cd.half_hour_evaporation(airtemp=at, rh=rh, airpress=ap, rs=rs, rext=rext, u=u, z=z)
        return half_hour_eo


class Checkdam_Parameters(object):
    def __init__(self,check_dam_name, catchment_area, rainfall, date_time_index, evaporation, infiltration_rate):
        self.check_dam_name = check_dam_name
        self.catchment_area = catchment_area
        self.rainfall = rainfall
        self.date_time_index = date_time_index
        self.evaporation = evaporation
        self.infiltration_rate = infiltration_rate
        self.duration = len(date_time_index)

    def simulate_inflow(self, catchment_area, rainfall, date_time_index):
        catchment_area = catchment_area or self.catchment_area
        rainfall = rainfall or self.rainfall
        date_time_index = date_time_index or self.date_time_index
        duration = len(date_time_index) or self.duration
        inflow = np.zeros(duration)


    def __repr__(self):
        return "Check dam no %s" %(self.check_dam_name)


class Checkdam_routing(object):
    def __init__(self, inflow, infiltration, evaporation, current_volume, max_volume):
        self.inflow = inflow
        self.infiltration = infiltration
        self.evaporation = evaporation
        self.current_volume = current_volume
        self.max_volume = max_volume
        self.current_volume, self.overflow = self.routing()

    def routing(self):
        self.current_volume = (self.current_volume + self.inflow) - (self.infiltration + self.evaporation)
        if self.current_volume > self.max_volume:
            self.overflow = self.current_volume - self.max_volume
            self.current_volume = self.max_volume
        else:
            self.overflow = 0.0
        return self.current_volume, self.overflow







# class Checkdam(Checkdam_Parameters):
#     def __init__(self, max_volume, stage_area, stage_volume, initial_volume, catchment_area):
#         self.max_volume = max_volume
#         self.stage_area = stage_area
#         self.stage_volume = stage_volume
#         self.initial_volume = initial_volume
#         self.catchment_area = catchment_area




"""
check dam chain
http://stackoverflow.com/a/2482610/2632856
"""

ch_463_lat = 13.360354
ch_463_long = 77.527267

"""
Open water evaporation 463
"""
weather_df['Average Temp (C)'] = 0.5*(weather_df['Min Air Temperature (C)'] + weather_df['Max Air Temperature (C)'])
weather_df['Solar Radiation (MJ/m2/30min)'] = (weather_df['Solar Radiation (Wpm2)'] * 1800)/(10**6)
airtemp = weather_df['Average Temp (C)']
hum = weather_df['Humidity (%)']
rs = weather_df['Solar Radiation (MJ/m2/30min)']
wind_speed = weather_df['Wind Speed (mps)']

weather_463 = Open_Water_Evaporation(check_dam_name="463",air_temperature=airtemp, relative_humidity=hum, incoming_solar_radiation=rs, wind_speed_mps=wind_speed,elevation=799, date_time_index=weather_df.index, latitdude=ch_463_lat, longitude=ch_463_long)
weather_463 = weather_463.calculate_half_hour_EO()
weather_463_df = pd.DataFrame(weather_463, index=weather_df.index, columns=['Evaporation (mm)'])
weather_463_df = weather_463_df.join(rain_df, how='right')
weather_463_df_daily = weather_463_df.resample('D', how=np.sum)
print(weather_463_df_daily.head())
