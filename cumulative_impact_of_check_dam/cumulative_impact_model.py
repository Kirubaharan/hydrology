#!/usr/bin/env python
__author__ = 'kiruba'
""" This script is an attempt to model the cumulative impacts of check dam"""
import sys
# sys.path.append('../checkdam/')
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
# import checkdam as cd
from datetime import timedelta


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



@contextmanager
def columns_in(dataframe):
    """
    Source: https://gist.github.com/aflaxman/4121076#file-sdm_diaper_delivery-ipynb
    :param dataframe: Pandas dataframe
    :return: column names in df as a tuple
    """
    col_names = dataframe.columns
    col_list = [dataframe[col] for col in col_names]
    try:
        yield tuple(col_list)
    finally:
        for i, col, in enumerate(col_names):
            dataframe[col] = col_list[i]

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
# print weather_df.head()

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
        self.half_hour_eo = self.calculate_half_hour_eo()

    def calculate_air_pressure(self, elevation=None):    # None is the key here
        z = elevation or self.elevation
        p = ((1 - (2.25577 * (10 ** -5) * z)))
        air_p_pa = 101325 * (p ** 5.25588)
        return air_p_pa

    def calculate_extraterrestrial_irradiation(self, date_time=None, latitude=None, longitude=None):
        lat = latitude or self.latitude
        lon = longitude or self.longitude
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
            lm = (180 + (180 - lon))  # longitude of measurement site
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
                lm = (180 + (180 - lon))  # longitude of measurement site
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

    def calculate_half_hour_eo(self, airtemp=None, rh=None, airpress=None, rs=None, rext=None,u=None, z=None):
        at = airtemp or self.air_temperature
        rh = rh or self.relative_humidity
        ap = airpress or self.air_pressure
        rs = rs or self.incoming_solar_radiation
        rext = rext or self.extraterrestrial_irradiation
        u = u or self.wind_speed_mps
        z = z or self.elevation
        half_hour_eo = cd.half_hour_evaporation(airtemp=at, rh=rh, airpress=ap, rs=rs, rext=rext, u=u, z=z)
        return half_hour_eo




class CheckdamParameters(object):
    def __init__(self, check_dam_name, catchment_area, evaporation, infiltration_rate, max_volume, stage_volume_csv, stage_area_csv, own_catchment_inflow_ratio, previous_check_dam=None, next_check_dam=None, initial_volume=None):
        self.check_dam_name = check_dam_name
        self.catchment_area = catchment_area
        self.evaporation = evaporation   # enter in mm/day
        self.infiltration_rate = infiltration_rate # enter in m/day
        self.max_volume = max_volume
        self.initial_volume = 0.0 or initial_volume
        self.stage_cutoff = 0.1  # in meter constant
        self.previous_check_dam = previous_check_dam
        self.next_check_dam = next_check_dam
        self.own_catchment_inflow_ratio = own_catchment_inflow_ratio
        # self.stage_volume = stage_volume_csv
        # self.stage_area = stage_area_csv
        self.stage_volume_df = self.convert_stage_volume_csv_to_df(stage_volume_csv)
        self.stage_area_df = self.convert_stage_area_csv_to_df(stage_area_csv)

    def convert_stage_volume_csv_to_df(self, csv):
        df = pd.read_csv(csv, sep=',', header=0, names=['sno', 'stage_m', 'volume_cu_m'])   # x stage y= volume/area
        df.drop('sno', inplace=True, axis=1)
        df.set_index(df['volume_cu_m'], inplace=True)
        return df

    def convert_stage_area_csv_to_df(self, csv):
        df = pd.read_csv(csv, sep=',', header=0, names=['sno', 'stage_m', 'area_sq_m'])
        df.drop('sno', inplace=True, axis=1)
        df.set_index(df['stage_m'], inplace=True)
        return df

    def convert_volume_to_area(self, volume):
        vol_1, vol_2 = cd.find_range(self.stage_volume_df['volume_cu_m'].tolist(), volume) #x
        stage_1 = self.stage_volume_df.loc[vol_1, 'stage_m'] # y
        stage_2 = self.stage_volume_df.loc[vol_2, 'stage_m']
        slope_vol = (stage_2 - stage_1) / (vol_2 - vol_1)
        intercept_vol = stage_2 - (slope_vol*vol_2)
        obs_stage = (slope_vol*volume) + intercept_vol
        if obs_stage >= self.stage_cutoff:
            stage_1, stage_2 = cd.find_range(self.stage_area_df['stage_m'].tolist(), obs_stage)
            area_1 = self.stage_area_df.loc[stage_1, 'area_sq_m']
            area_2 = self.stage_area_df.loc[stage_2, 'area_sq_m']
            slope_area = (stage_2 - stage_1) / (area_2 - area_1)
            intercept_area = area_2 - (slope_area *  stage_2)
            area = (slope_area * obs_stage) + intercept_area
        else:
            area = 0.0
        return area

    # def convert_stage_to_volume(self, stage):




    def __repr__(self):
        return "Check dam no %s" % self.check_dam_name


class CheckdamRouting(object):
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


class CheckdamChain(object):
    def __init__(self, inflow_catchment_area_df, check_dam_chain, slope, intercept):
        self.inflow_catchment_area_df = inflow_catchment_area_df
        self.check_dam_chain = check_dam_chain # list of check dams in order, must be instance of CheckdamParameters class
        self.slope = slope
        self.intercept = intercept
        self.duration = len(self.inflow_catchment_area_df.index)
        self.no_of_check_dams = len(self.check_dam_chain)
        self.output_df = self.create_output_df()

    def create_output_df(self):
        output_df = self.inflow_catchment_area_df
        for checkdam in self.check_dam_chain:
            if not isinstance(checkdam, CheckdamParameters):
                raise TypeError("{0} is not an instance of CheckdamParameters()".format(checkdam.check_dam_name))
            output_df[('volume_{0:d}'.format(checkdam.check_dam_name))] = 0.0
            output_df[('inflow_{0:d}'.format(checkdam.check_dam_name))] = 0.0
            output_df[('evap_{0:d}'.format(checkdam.check_dam_name))] = 0.0
            output_df[('infilt_{0:d}'.format(checkdam.check_dam_name))] = 0.0
            output_df[('overflow_{0:d}'.format(checkdam.check_dam_name))] = 0.0
            output_df[('conv_ratio_{0:d}'.format(checkdam.check_dam_name))] = 0.0
            output_df[('est_own_flow_ratio_{0:d}'.format(checkdam.check_dam_name))] = 0.0
            # assign initial volume
            output_df[('volume_{0:d}'.format(checkdam.check_dam_name))][0] = checkdam.initial_volume
        return output_df

    @property
    def simulate(self):
        last_date = max(self.output_df.index)
        for dt in self.output_df.index:
            if dt < last_date:
                # print dt
                for checkdam in self.check_dam_chain:
                    # print checkdam.check_dam_name
                    if not isinstance(checkdam, CheckdamParameters):
                        raise TypeError("{0} is not an instance of CheckdamParameters()".format(checkdam.checkdam_name))
                    #inflow calculation
                    n = 1
                    convergence_ratio = 5
                    assumed_own_catchment_inflow_ratio = checkdam.own_catchment_inflow_ratio
                    while True:
                        # print n, convergence_ratio
                        if (round(convergence_ratio, 2) == round(1.0, 2)) or (n > 100) or (round(convergence_ratio, 2) == round(0.00, 2)):    # http://stackoverflow.com/a/14928206/2632856
                            self.output_df.loc[dt, 'conv_ratio_{0:d}'.format(checkdam.check_dam_name)] = convergence_ratio
                            convergence_ratio = 5
                            break
                        else:
                            inflow_from_catchment_area_ratio = (self.output_df.loc[dt,'diff'] * self.slope) + self.intercept
                            inflow_from_catchment = (checkdam.catchment_area* inflow_from_catchment_area_ratio * assumed_own_catchment_inflow_ratio)
                            inflow_from_overflow =  self.output_df.loc[dt, ('inflow_{0:d}'.format(checkdam.check_dam_name))]
                            # print 'inflow = ', inflow_from_catchment
                            if inflow_from_catchment > 0.0:
                                # print "ok"
                                estimated_own_catchment_inflow_ratio = inflow_from_catchment / (inflow_from_catchment + inflow_from_overflow)
                                convergence_ratio = estimated_own_catchment_inflow_ratio / assumed_own_catchment_inflow_ratio
                                assumed_own_catchment_inflow_ratio = 0.5 * (estimated_own_catchment_inflow_ratio + assumed_own_catchment_inflow_ratio)
                                print 'convergence ratio = {0:.2f} for iteration {1:d}'.format(convergence_ratio, n)
                                inflow = inflow_from_catchment + inflow_from_overflow
                                n += 1
                            else:
                                estimated_own_catchment_inflow_ratio = 0.0
                                inflow = inflow_from_overflow
                                convergence_ratio = 5
                                n = 101
                                # print "ok" * 10

                    # print inflow
                    self.output_df.loc[dt, ('inflow_{0:d}'.format(checkdam.check_dam_name))] = inflow
                    # calculate surface area
                    surface_area = checkdam.convert_volume_to_area(self.output_df.loc[dt, ('volume_{0:d}'.format(checkdam.check_dam_name))])
                    # calculate evaporationprint 'ok'
                    evaporation = surface_area * checkdam.evaporation.loc[dt] * 0.001
                    # print evaporation, surface_area, checkdam.evaporation.loc[dt]
                    # assign evaporation
                    self.output_df.loc[dt, ('evap_{0:d}'.format(checkdam.check_dam_name))] = evaporation
                    # calculate infiltration
                    infiltration = surface_area * checkdam.infiltration_rate
                    self.output_df.loc[dt, ('infilt_{0:d}'.format(checkdam.check_dam_name))] = infiltration
                    # assign current volume
                    current_volume = self.output_df.loc[dt, ('volume_{0:d}'.format(checkdam.check_dam_name))]
                    # do routing
                    # print inflow, evaporation, infiltration, current_volume, checkdam.max_volume
                    checkdam_routing = CheckdamRouting(inflow=inflow, evaporation=evaporation, infiltration=infiltration, current_volume=current_volume, max_volume=checkdam.max_volume )
                    self.output_df.loc[[dt, dt + timedelta(days=1)], ('volume_{0:d}'.format(checkdam.check_dam_name))] = checkdam_routing.current_volume
                    # print evaporation, surface_area, checkdam.evaporation.loc[dt], checkdam_routing.current_volume
                    if (checkdam_routing.overflow > 0.0):
                        # print "overflow" * 10
                        self.output_df.loc[dt, ('overflow_{0:d}'.format(checkdam.check_dam_name))] = checkdam_routing.overflow
                        if not checkdam.next_check_dam is None:
                            self.output_df.loc[dt+timedelta(days=1), ('inflow_{0:d}'.format(checkdam.next_check_dam))] = checkdam_routing.overflow
        return self.output_df
                        # self.output_df.loc[dt, ('overflow_{0:d}'.format(checkdam.check_dam_name))] = checkdam_routing.overflow
                        # add overflow from dt to dt + 1 day's inflow
                        # if isinstance(checkdam.next_check_dam, CheckdamParameters):
                        #     self.output_df.loc[dt+timedelta(days=1), ('inflow_{0:d}'.format(checkdam.next_check_dam.check_dam_name))] = checkdam_routing.overflow
                    # else:
                    #      self.output_df.loc[[dt,dt+timedelta(days=1)], ('volume_{0:d}'.format(checkdam.check_dam_name))] = checkdam_routing.current_volume




# create pandas df, index , data with inflow catchment, create columns for each check dam attribute and assign

# check dam chain
# http://stackoverflow.com/a/2482610/2632856


date_format = '%Y-%m-%d %H:%M:%S'
daily_format = '%Y-%m-%d'
# Weather file
weather_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/hadonahalli/corrected_weather_ws.csv'
# Rain file
rain_file = '/media/kiruba/New Volume/milli_watershed/cumulative impacts/had_rainfall_daily.csv'
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
print rain_df.head()
# set index
rain_df['Date_Time'] = pd.to_datetime(rain_df['date_time'], format=daily_format)
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


"""
stage vs volume / stage vs area
"""
ch_463_stage_area_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_463/cont_area.csv'
ch_463_stage_area_df = pd.read_csv(ch_463_stage_area_file, sep=',', header=0)
ch_463_stage_volume_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_463/stage_vol.csv'
ch_463_stage_volume_df = pd.read_csv(ch_463_stage_volume_file, sep=',', header=0)
default_stage_area_file = '/media/kiruba/New Volume/milli_watershed/default_stage_area.csv'
default_stage_area_df = pd.read_csv(default_stage_area_file, sep=',', header=0)
default_stage_volume_file = '/media/kiruba/New Volume/milli_watershed/default_stage_volume.csv'
default_stage_volume_df = pd.read_csv(default_stage_volume_file, sep=',', header=0)
# print ch_463_stage_area_df
# print ch_463_stage_volume_df

# raise SystemExit(0)

"""
inflow per catchment area ratio
"""
catchment_area_634 = 0.145   # sq.km
inflow_catchment_area_had_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_634/et_infilt_634_w_of.csv'
inflow_catchment_area_had_df = pd.read_csv(inflow_catchment_area_had_file, sep=',', header=0)
inflow_catchment_area_had_df['Date'] = pd.to_datetime(inflow_catchment_area_had_df['Date'], format=daily_format)
inflow_catchment_area_had_df.set_index(inflow_catchment_area_had_df['Date'], inplace=True)
inflow_catchment_area_had_df.sort_index(inplace=True)
inflow_catchment_area_had_df.drop('Date', 1, inplace=True)

# divide inflow by catchment
inflow_catchment_area_had_df['inflow_catchment_area_ratio'] = inflow_catchment_area_had_df['Inflow (cu.m)'] / catchment_area_634
# inflow_catchment_area_had_df = inflow_catchment_area_had_df.loc[:, ['inflow_catchment_area_ratio']]
# join rainfall
inflow_catchment_area_had_df = inflow_catchment_area_had_df.join(rain_df['diff'], how='left')
# print rain_df['rain(mm)']['2014-05-17']
print inflow_catchment_area_had_df.head()
# print inflow_catchment_area_had_df.tail()
ch_463_lat = 13.360354
ch_463_long = 77.527267
# relationship between rainfall and inflow per catchment area
x_cal = inflow_catchment_area_had_df['diff']
y_cal = inflow_catchment_area_had_df['inflow_catchment_area_ratio']
inflow_catchment_area_rain = cd.polyfit(x_cal, y_cal, 1)
coeff_cal = inflow_catchment_area_rain['polynomial']
slope = coeff_cal[0]
intercept = coeff_cal[1]
r_squared = inflow_catchment_area_rain['determination']
x_cal_new = np.linspace(min(x_cal), max(x_cal), 50)
polynomial = np.poly1d(coeff_cal)
y_cal_new = polynomial(x_cal_new)
print r_squared
fig = plt.figure()
# plt.plot(inflow_catchment_area_had_df.index, inflow_catchment_area_had_df['diff'], 'o')
plt.plot( inflow_catchment_area_had_df['diff'], inflow_catchment_area_had_df['inflow_catchment_area_ratio'], 'ro')
plt.plot(x_cal_new, y_cal_new, 'g-')
plt.xlim(-10, 100)
plt.ylim(-10, 100)
plt.show()

# log_x_cal = np.log(x_cal)
# log_y_cal = np.log(y_cal)
#
# OK = log_y_cal == log_y_cal
# masked_log_y_cal = log_y_cal[OK]
# masked_log_x_cal = log_x_cal[OK]
# fig = plt.figure()
# plt.plot(inflow_catchment_area_had_df.index, inflow_catchment_area_had_df['Inflow (cu.m)'], 'o')
# plt.plot(inflow_catchment_area_had_df.index, inflow_catchment_area_had_df['diff'], 'ro')
# plt.plot(masked_log_x_cal, masked_log_y_cal, 'ro')
# plt.show()
# print rain_df.head()
raise SystemExit(0)



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
weather_463 = weather_463.calculate_half_hour_eo()
weather_463_df = pd.DataFrame(weather_463, index=weather_df.index, columns=['Evaporation (mm)'])
# weather_463_df = weather_463_df.join(rain_df, how='right')
weather_463_df_daily = weather_463_df.resample('D', how=np.sum)
weather_463_df_daily = weather_463_df_daily.join(rain_df, how='left')
# print(weather_463_df_daily.head())

"""
Parametrisation of the model
"""
# common for hadonahalli
rainfall_had = rain_df['rain (mm)']
time_duration_had = inflow_catchment_area_had_df.index
infiltration_rate = 0.002 # m/day
# 463
catchment_area_463 = 0.182  # dummy value as of now
evaporation_463 = weather_463_df_daily['Evaporation (mm)']
# print evaporation_463
max_volume_463 = 5.0
stage_volume_463 = ch_463_stage_volume_file
stage_area_463 = ch_463_stage_area_file
own_catchment_inflow_ratio_463 = 1.0
checkdam_463 = CheckdamParameters(check_dam_name=463, catchment_area=catchment_area_463, infiltration_rate=infiltration_rate, evaporation=evaporation_463, max_volume=max_volume_463, stage_volume_csv=stage_volume_463, stage_area_csv=stage_area_463,previous_check_dam=None, own_catchment_inflow_ratio=own_catchment_inflow_ratio_463)
checkdam_463.initial_volume = 0.0


# 640
catchment_area_640 = 0.07  # dummy value as of now
evaporation_640 = weather_463_df_daily['Evaporation (mm)']
max_volume_640 = 5.0
stage_volume_640 = ch_463_stage_volume_file
stage_area_640 = ch_463_stage_area_file
own_catchment_inflow_ratio_640 = 1.0


checkdam_640 = CheckdamParameters(check_dam_name=640, catchment_area=catchment_area_640, infiltration_rate=infiltration_rate, evaporation=evaporation_640, max_volume=max_volume_640, stage_volume_csv=stage_volume_640, stage_area_csv=stage_area_640,previous_check_dam=checkdam_463, own_catchment_inflow_ratio=own_catchment_inflow_ratio_640)
checkdam_640.initial_volume = 0.0  # check for assigning correct initial volume
checkdam_463.next_check_dam = checkdam_640.check_dam_name
# print type(checkdam_640.check_dam_name)
had_chain_1 = CheckdamChain(inflow_catchment_area_df=inflow_catchment_area_had_df, check_dam_chain=[checkdam_463, checkdam_640], slope=slope, intercept=intercept)
had_chain_1.create_output_df()
had_output_1_df = had_chain_1.simulate
# print had_output_df.head()
#test plots

for checkdam in had_chain_1.check_dam_chain:
    fig = plt.figure()
    plt.plot(had_chain_1.output_df.index, had_chain_1.output_df[('inflow_{0:d}'.format(checkdam.check_dam_name))], '-', label=('inflow_{0:d}'.format(checkdam.check_dam_name)))
    plt.plot(had_chain_1.output_df.index, had_chain_1.output_df[('evap_{0:d}'.format(checkdam.check_dam_name))], '-', label=('evap_{0:d}'.format(checkdam.check_dam_name)))
    plt.plot(had_chain_1.output_df.index, had_chain_1.output_df[('volume_{0:d}'.format(checkdam.check_dam_name))], '-', label=('volume_{0:d}'.format(checkdam.check_dam_name)))
    plt.plot(had_chain_1.output_df.index, had_chain_1.output_df[('infilt_{0:d}').format(checkdam.check_dam_name)], '-', label=('infilt_{0:d}'.format(checkdam.check_dam_name)))
    plt.plot(had_chain_1.output_df.index, had_chain_1.output_df[('overflow_{0:d}'.format(checkdam.check_dam_name))], '-', label=('overflow_{0:d}'.format(checkdam.check_dam_name)))
    plt.legend()
plt.show()
had_output_1_df.to_csv('/media/kiruba/New Volume/milli_watershed/cumulative impacts/had_chain_1.csv')
