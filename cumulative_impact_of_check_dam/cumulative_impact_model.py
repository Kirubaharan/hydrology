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
    def __init__(self, check_dam_name, catchment_area, evaporation, infiltration_rate, max_height, stage_volume_csv, stage_area_csv, own_catchment_inflow_ratio, previous_check_dam=None, next_check_dam=None, initial_volume=None):
        self.check_dam_name = check_dam_name
        self.catchment_area = catchment_area
        self.evaporation = evaporation   # enter in mm/day
        self.infiltration_rate = infiltration_rate # enter in m/day
        self.max_height = max_height
        self.initial_volume = 0.0 or initial_volume
        self.stage_cutoff = 0.1  # in meter constant
        self.previous_check_dam = previous_check_dam
        self.next_check_dam = next_check_dam
        self.own_catchment_inflow_ratio = own_catchment_inflow_ratio
        # self.stage_volume = stage_volume_csv
        # self.stage_area = stage_area_csv
        self.stage_volume_df = self.convert_stage_volume_csv_to_df(stage_volume_csv)
        self.stage_area_df = self.convert_stage_area_csv_to_df(stage_area_csv)
        self.stage_volume_df_indexed_by_stage = self.convert_stage_volume_with_stage_index_csv_to_df(stage_volume_csv)
        self.max_volume = self.estimate_max_volume(max_height=self.max_height)

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

    def convert_stage_volume_with_stage_index_csv_to_df(self,csv):
        df = pd.read_csv(csv, sep=',', header=0, names=['sno', 'stage_m', 'volume_cu_m'])   # x stage y= volume/area
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

    def estimate_max_volume(self, max_height=0.0):
        max_height = self.max_height or max_height
        if max_height > 0.0:
            stage_1, stage_2 = cd.find_range(self.stage_volume_df_indexed_by_stage['stage_m'].tolist(), max_height)
            vol_1 = self.stage_volume_df_indexed_by_stage.loc[stage_1, 'volume_cu_m']
            vol_2 = self.stage_volume_df_indexed_by_stage.loc[stage_2, 'volume_cu_m']
            slope_stage = (vol_2 - vol_1) / (stage_2 - stage_1)
            intercept_stage = vol_2 - (slope_stage * stage_2)
            max_volume = (slope_stage*max_height) + intercept_stage
        else:
            max_volume = 0.0
        return max_volume




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
    def __init__(self, inflow_catchment_area_df, check_dam_chain, slope, intercept, output_df=None):
        self.inflow_catchment_area_df = inflow_catchment_area_df
        self.check_dam_chain = check_dam_chain # list of check dams in order, must be instance of CheckdamParameters class
        self.slope = slope
        self.intercept = intercept
        self.duration = len(self.inflow_catchment_area_df.index)
        self.no_of_check_dams = len(self.check_dam_chain)
        self.checkdam_list = self.create_check_dam_list()
        if output_df is None:
            self.output_df = self.create_output_df()
        else:
            self.modify_df = output_df
            self.output_df = self.modify_output_df()

    def create_check_dam_list(self):
        checkdam_list = []
        for checkdam in self.check_dam_chain:
            checkdam_list.append(checkdam)
        return checkdam_list

    def create_output_df(self):
        output_df = self.inflow_catchment_area_df
        for checkdam in self.check_dam_chain:
            if not isinstance(checkdam, CheckdamParameters):
                raise TypeError("{0} is not an instance of CheckdamParameters()".format(checkdam.check_dam_name))
            output_df[('volume_{0:d}'.format(checkdam.check_dam_name))] = 0.0
            output_df[('inflow_from_catchment_{0:d}'.format(checkdam.check_dam_name))] = 0.0
            output_df[('inflow_from_overflow_{0:d}'.format(checkdam.check_dam_name))] = 0.0
            output_df[('total_inflow_{0:d}'.format(checkdam.check_dam_name))] = 0.0
            output_df[('evap_{0:d}'.format(checkdam.check_dam_name))] = 0.0
            output_df[('infilt_{0:d}'.format(checkdam.check_dam_name))] = 0.0
            output_df[('overflow_{0:d}'.format(checkdam.check_dam_name))] = 0.0
            output_df[('conv_ratio_{0:d}'.format(checkdam.check_dam_name))] = 0.0
            output_df[('est_own_flow_ratio_{0:d}'.format(checkdam.check_dam_name))] = 0.0
            # assign initial volume
            output_df[('volume_{0:d}'.format(checkdam.check_dam_name))][0] = checkdam.initial_volume
        return output_df

    def modify_output_df(self):
        output_df = self.modify_df
        for checkdam in self.check_dam_chain:
            if not isinstance(checkdam, CheckdamParameters):
                raise TypeError("{0} is not an instance of CheckdamParameters()".format(checkdam.check_dam_name))
            output_df[('volume_{0:d}'.format(checkdam.check_dam_name))] = 0.0
            output_df[('inflow_from_catchment_{0:d}'.format(checkdam.check_dam_name))] = 0.0
            # output_df[('inflow_from_overflow_{0:d}'.format(checkdam.check_dam_name))] = 0.0
            output_df[('total_inflow_{0:d}'.format(checkdam.check_dam_name))] = 0.0
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
                            self.output_df.loc[dt, ('inflow_from_catchment_{0:d}'.format(checkdam.check_dam_name))] = inflow_from_catchment
                            inflow_from_overflow =  self.output_df.loc[dt, ('inflow_from_overflow_{0:d}'.format(checkdam.check_dam_name))]
                            # print 'inflow = ', inflow_from_catchment
                            if inflow_from_catchment > 0.0:
                                # print "ok"
                                estimated_own_catchment_inflow_ratio = inflow_from_catchment / (inflow_from_catchment + inflow_from_overflow)
                                convergence_ratio = estimated_own_catchment_inflow_ratio / assumed_own_catchment_inflow_ratio
                                assumed_own_catchment_inflow_ratio = 0.5 * (estimated_own_catchment_inflow_ratio + assumed_own_catchment_inflow_ratio)
                                # print 'convergence ratio = {0:.2f} for iteration {1:d}'.format(convergence_ratio, n)
                                inflow = inflow_from_catchment + inflow_from_overflow
                                n += 1
                            else:
                                estimated_own_catchment_inflow_ratio = 0.0
                                inflow = inflow_from_overflow
                                convergence_ratio = 5
                                n = 101
                                # print "ok" * 10

                    # print inflow
                    self.output_df.loc[dt, ('total_inflow_{0:d}'.format(checkdam.check_dam_name))] = inflow
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
                            self.output_df.loc[dt+timedelta(days=1), ('inflow_from_overflow_{0:d}'.format(checkdam.next_check_dam))] = checkdam_routing.overflow
        return self.output_df
                        # self.output_df.loc[dt, ('overflow_{0:d}'.format(checkdam.check_dam_name))] = checkdam_routing.overflow
                        # add overflow from dt to dt + 1 day's inflow
                        # if isinstance(checkdam.next_check_dam, CheckdamParameters):
                        #     self.output_df.loc[dt+timedelta(days=1), ('inflow_{0:d}'.format(checkdam.next_check_dam.check_dam_name))] = checkdam_routing.overflow
                    # else:
                    #      self.output_df.loc[[dt,dt+timedelta(days=1)], ('volume_{0:d}'.format(checkdam.check_dam_name))] = checkdam_routing.current_volume


class CheckdamChainMerge(object):
    def __init__(self, checkdam_chain_a, checkdam_chain_b, check_dam_a, check_dam_b, converging_check_dam, merged_df=None):
        self.checkdam_chain_a = checkdam_chain_a
        self.checkdam_chain_b = checkdam_chain_b
        self.check_dam_a = check_dam_a
        self.check_dam_b = check_dam_b
        self.converging_check_dam = converging_check_dam
        self.output_df_a = self.checkdam_chain_a.output_df
        self.output_df_b = self.checkdam_chain_b.output_df
        self.checkdam_list = self.merge_check_dam_list()
        if merged_df is None:
            self.output_df = self.merge_df(df_a=self.output_df_a, df_b=self.output_df_b)
        else:
            self.output_df = merged_df

    def merge_df(self, df_a, df_b):
        # columns_to_use = df_a.columns.difference(df_b.columns)
        # merged_df = df_a.merge(df_b, left_index=True, right_index=True, how='outer')\
        merged_df = df_a.join
        return merged_df

    def merge_check_dam_list(self):
        merged_list = self.checkdam_chain_a.checkdam_list + self.checkdam_chain_b.checkdam_list
        merged_list.append(self.converging_check_dam)
        return merged_list

    @property
    def simulate(self):
        checkdam = self.converging_check_dam
        self.output_df[('volume_{0:d}'.format(checkdam.check_dam_name))] = 0.0
        self.output_df[('inflow_from_catchment_{0:d}'.format(checkdam.check_dam_name))] = 0.0
        self.output_df[('inflow_from_overflow_{0:d}'.format(checkdam.check_dam_name))] = 0.0
        self.output_df[('total_inflow_{0:d}'.format(checkdam.check_dam_name))] = 0.0
        self.output_df[('evap_{0:d}'.format(checkdam.check_dam_name))] = 0.0
        self.output_df[('infilt_{0:d}'.format(checkdam.check_dam_name))] = 0.0
        self.output_df[('overflow_{0:d}'.format(checkdam.check_dam_name))] = 0.0
        self.output_df[('conv_ratio_{0:d}'.format(checkdam.check_dam_name))] = 0.0
        self.output_df[('est_own_flow_ratio_{0:d}'.format(checkdam.check_dam_name))] = 0.0
        # add two previous check dam's overflow to get inflow from check dam
        self.output_df[('inflow_from_overflow_{0:d}'.format(checkdam.check_dam_name))] = self.output_df_a[('overflow_{0:d}'.format(self.check_dam_a))] + self.output_df_b[('overflow_{0:d}'.format(self.check_dam_b))]
        last_date = max(self.output_df.index)
        for dt in self.output_df.index:
            if dt < last_date:
                n = 1
                convergence_ratio = 5
                assumed_own_catchment_inflow_ratio = self.converging_check_dam.own_catchment_inflow_ratio
                while True:
                    if (round(convergence_ratio, 2) == round(1.0, 2)) or (n > 100) or (round(convergence_ratio, 2) == round(0.00, 2)):
                        self.output_df.loc[dt, 'conv_ratio{0:d}'.format(checkdam.check_dam_name)] = convergence_ratio
                        convergence_ratio = 5
                        break
                    else:
                        inflow_from_catchment_area_ratio = (self.output_df.loc[dt, 'diff'] * 1.1868) + 0.508
                        inflow_from_catchment = (checkdam.catchment_area * inflow_from_catchment_area_ratio * assumed_own_catchment_inflow_ratio)
                        self.output_df.loc[dt, ('inflow_from_catchment_{0:d}'.format(checkdam.check_dam_name))] = inflow_from_catchment
                        inflow_from_overflow =  self.output_df.loc[dt, ('inflow_from_overflow_{0:d}'.format(checkdam.check_dam_name))]
                        # print 'inflow = ', inflow_from_catchment
                        if inflow_from_catchment > 0.0:
                            # print "ok"
                            estimated_own_catchment_inflow_ratio = inflow_from_catchment / (inflow_from_catchment + inflow_from_overflow)
                            convergence_ratio = estimated_own_catchment_inflow_ratio / assumed_own_catchment_inflow_ratio
                            assumed_own_catchment_inflow_ratio = 0.5 * (estimated_own_catchment_inflow_ratio + assumed_own_catchment_inflow_ratio)
                            # print 'convergence ratio = {0:.2f} for iteration {1:d}'.format(convergence_ratio, n)
                            inflow = inflow_from_catchment + inflow_from_overflow
                            n += 1
                        else:
                            estimated_own_catchment_inflow_ratio = 0.0
                            inflow = inflow_from_overflow
                            convergence_ratio = 5
                            n = 101
                            # print "ok" * 10

                # print inflow
                self.output_df.loc[dt, ('total_inflow_{0:d}'.format(checkdam.check_dam_name))] = inflow
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
                        self.output_df.loc[dt+timedelta(days=1), ('inflow_from_overflow_{0:d}'.format(checkdam.next_check_dam))] = checkdam_routing.overflow
        return self.output_df







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

rain_df['index'] = rain_df.index
rain_df.drop_duplicates(subset='index', take_last=True, inplace=True)
del rain_df['index']
rain_df = rain_df.sort()


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
# print inflow_catchment_area_had_df.head()
# print inflow_catchment_area_had_df['2014-05-14']
# raise SystemExit(0)
# print rain_df['rain(mm)']['2014-05-17']
# print inflow_catchment_area_had_df.head()
# print inflow_catchment_area_had_df.tail()
# relationship between rainfall and inflow per catchment area
x_cal = inflow_catchment_area_had_df['diff']
y_cal = inflow_catchment_area_had_df['inflow_catchment_area_ratio']
inflow_catchment_area_rain = cd.polyfit(x_cal, y_cal, 1)
coeff_cal = inflow_catchment_area_rain['polynomial']
slope = coeff_cal[0]
intercept = coeff_cal[1]
print "slope, intercept"
print slope
print intercept
# raise SystemExit(0)
r_squared = inflow_catchment_area_rain['determination']
x_cal_new = np.linspace(min(x_cal), max(x_cal), 50)
polynomial = np.poly1d(coeff_cal)
y_cal_new = polynomial(x_cal_new)
# print r_squared
# fig = plt.figure()
# plt.plot(inflow_catchment_area_had_df.index, inflow_catchment_area_had_df['diff'], 'o')
# plt.plot( inflow_catchment_area_had_df['diff'], inflow_catchment_area_had_df['inflow_catchment_area_ratio'], 'ro')
# plt.plot(x_cal_new, y_cal_new, 'g-')
# plt.xlim(-10, 100)
# plt.ylim(-10, 100)
# plt.show()

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
# raise SystemExit(0)



"""
Open water evaporation HAD
"""
weather_df['Average Temp (C)'] = 0.5*(weather_df['Min Air Temperature (C)'] + weather_df['Max Air Temperature (C)'])
weather_df['Solar Radiation (MJ/m2/30min)'] = (weather_df['Solar Radiation (Wpm2)'] * 1800)/(10**6)
airtemp = weather_df['Average Temp (C)']
hum = weather_df['Humidity (%)']
rs = weather_df['Solar Radiation (MJ/m2/30min)']
wind_speed = weather_df['Wind Speed (mps)']

"""
Parametrisation of the model
"""
"""
Chain 1 463 -> 640 -> 1
"""
# common for hadonahalli
rainfall_had = rain_df['diff']  # mm/day
time_duration_had = inflow_catchment_area_had_df.index
infiltration_rate = 0.002 # m/day
# 463
catchment_area_463 = 0.182
elevation_463 = 839
ch_463_lat = 13.360354
ch_463_long = 77.527267
weather_463 = Open_Water_Evaporation(check_dam_name="463",air_temperature=airtemp, relative_humidity=hum, incoming_solar_radiation=rs, wind_speed_mps=wind_speed,elevation=elevation_463, date_time_index=weather_df.index, latitdude=ch_463_lat, longitude=ch_463_long)
weather_463 = weather_463.calculate_half_hour_eo()
weather_463_df = pd.DataFrame(weather_463, index=weather_df.index, columns=['Evaporation (mm)'])
# weather_463_df = weather_463_df.join(rain_df, how='right')
weather_463_df_daily = weather_463_df.resample('D', how=np.sum)
weather_463_df_daily = weather_463_df_daily.join(rain_df, how='left')
evaporation_463 = weather_463_df_daily['Evaporation (mm)']
# print evaporation_463
max_height_463 = 0.7
stage_volume_463 = ch_463_stage_volume_file
stage_area_463 = ch_463_stage_area_file
own_catchment_inflow_ratio_463 = 1.0
checkdam_463 = CheckdamParameters(check_dam_name=463, catchment_area=catchment_area_463, infiltration_rate=infiltration_rate, evaporation=evaporation_463, max_height=max_height_463, stage_volume_csv=stage_volume_463, stage_area_csv=stage_area_463,previous_check_dam=None, own_catchment_inflow_ratio=own_catchment_inflow_ratio_463)
checkdam_463.initial_volume = 0.0
checkdam_463.next_check_dam = 640
""" ################################

# 640
catchment_area_640 = 0.07  # dummy value as of now
evaporation_640 = weather_463_df_daily['Evaporation (mm)']
max_height_640 = 0.5
elevation_640 = 838
ch_640_lat = 13.36007
ch_640_long = 77.52778
weather_640 = Open_Water_Evaporation(check_dam_name="640",air_temperature=airtemp, relative_humidity=hum, incoming_solar_radiation=rs, wind_speed_mps=wind_speed,elevation=elevation_640, date_time_index=weather_df.index, latitdude=ch_640_lat, longitude=ch_640_long)
weather_640 = weather_640.calculate_half_hour_eo()
weather_640_df = pd.DataFrame(weather_640, index=weather_df.index, columns=['Evaporation (mm)'])
weather_640_df_daily = weather_640_df.resample('D', how=np.sum)
weather_640_df_daily = weather_640_df_daily.join(rain_df, how='left')
evaporation_640 = weather_640_df_daily['Evaporation (mm)']
stage_volume_640 = default_stage_volume_file
stage_area_640 = default_stage_volume_file
own_catchment_inflow_ratio_640 = 1.0
checkdam_640 = CheckdamParameters(check_dam_name=640, catchment_area=catchment_area_640, infiltration_rate=infiltration_rate, evaporation=evaporation_640, max_height=max_height_640, stage_volume_csv=stage_volume_640, stage_area_csv=stage_area_640,previous_check_dam=checkdam_463, own_catchment_inflow_ratio=own_catchment_inflow_ratio_640)
checkdam_640.initial_volume = 0.0  # check for assigning correct initial volume
checkdam_640.next_check_dam = 1

# 1
catchment_area_1 = 0.8
evaporation_1 = weather_463_df_daily['Evaporation (mm)']
max_height_1 = 0.0
stage_volume_1 = default_stage_volume_file
stage_area_1 = default_stage_area_file
own_catchment_inflow_ratio_1 = 1.0
checkdam_1 = CheckdamParameters(check_dam_name=1, catchment_area=catchment_area_1, infiltration_rate=infiltration_rate, evaporation=evaporation_1, max_height=max_height_1, stage_volume_csv=stage_volume_1, stage_area_csv=stage_area_1, previous_check_dam=checkdam_640, own_catchment_inflow_ratio=own_catchment_inflow_ratio_1, )
checkdam_1.initial_volume = 0.0
checkdam_1.next_check_dam = 3
# print type(checkdam_640.check_dam_name)
# print inflow_catchment_area_had_df.head()
# raise SystemExit(0)
had_chain_1 = CheckdamChain(inflow_catchment_area_df=inflow_catchment_area_had_df, check_dam_chain=[checkdam_463, checkdam_640, checkdam_1], slope=slope, intercept=intercept)
had_chain_1.create_output_df()
had_output_1_df = had_chain_1.simulate


"""
# Chain 2 639 -> 641 -> 2
"""
# 639
catchment_area_639 = 0.624
elevation_639 = 831
ch_639_lat = 13.35314
ch_639_long = 77.53556
weather_639 = Open_Water_Evaporation(check_dam_name="639",air_temperature=airtemp, relative_humidity=hum, incoming_solar_radiation=rs, wind_speed_mps=wind_speed,elevation=elevation_639, date_time_index=weather_df.index, latitdude=ch_639_lat, longitude=ch_639_long)
weather_639 = weather_639.calculate_half_hour_eo()
weather_639_df = pd.DataFrame(weather_639, index=weather_df.index, columns=['Evaporation (mm)'])
# weather_639_df = weather_639_df.join(rain_df, how='right')
weather_639_df_daily = weather_639_df.resample('D', how=np.sum)
weather_639_df_daily = weather_639_df_daily.join(rain_df, how='left')
evaporation_639 = weather_639_df_daily['Evaporation (mm)']
# print evaporation_639
max_height_639 = 0.6
stage_volume_639 = default_stage_volume_file
stage_area_639 = default_stage_area_file
own_catchment_inflow_ratio_639 = 1.0
checkdam_639 = CheckdamParameters(check_dam_name=639, catchment_area=catchment_area_639, infiltration_rate=infiltration_rate, evaporation=evaporation_639, max_height=max_height_639, stage_volume_csv=stage_volume_639, stage_area_csv=stage_area_639,previous_check_dam=None, own_catchment_inflow_ratio=own_catchment_inflow_ratio_639)
checkdam_639.next_check_dam = 641
checkdam_639.initial_volume = 0.0

# 641
catchment_area_641 = 0.082
elevation_641 = 828
ch_641_lat = 13.35405
ch_641_long = 77.53615
weather_641 = Open_Water_Evaporation(check_dam_name="641",air_temperature=airtemp, relative_humidity=hum, incoming_solar_radiation=rs, wind_speed_mps=wind_speed,elevation=elevation_641, date_time_index=weather_df.index, latitdude=ch_641_lat, longitude=ch_641_long)
weather_641 = weather_641.calculate_half_hour_eo()
weather_641_df = pd.DataFrame(weather_641, index=weather_df.index, columns=['Evaporation (mm)'])
# weather_641_df = weather_641_df.join(rain_df, how='right')
weather_641_df_daily = weather_641_df.resample('D', how=np.sum)
weather_641_df_daily = weather_641_df_daily.join(rain_df, how='left')
evaporation_641 = weather_641_df_daily['Evaporation (mm)']
# print evaporation_641
max_height_641 = 0.5
stage_volume_641 = default_stage_volume_file
stage_area_641 = default_stage_area_file
own_catchment_inflow_ratio_641 = 1.0
checkdam_641 = CheckdamParameters(check_dam_name=641, catchment_area=catchment_area_641, infiltration_rate=infiltration_rate, evaporation=evaporation_641, max_height=max_height_641, stage_volume_csv=stage_volume_641, stage_area_csv=stage_area_641,previous_check_dam=checkdam_639, own_catchment_inflow_ratio=own_catchment_inflow_ratio_641)
checkdam_641.initial_volume = 0.0

# 2
catchment_area_2 = 0.152
# elevation_2 = 831
# ch_2_lat = 13.35314
# ch_2_long = 77.53556
# weather_2 = Open_Water_Evaporation(check_dam_name="2",air_temperature=airtemp, relative_humidity=hum, incoming_solar_radiation=rs, wind_speed_mps=wind_speed,elevation=elevation_2, date_time_index=weather_df.index, latitdude=ch_2_lat, longitude=ch_2_long)
# weather_2 = weather_2.calculate_half_hour_eo()
# weather_2_df = pd.DataFrame(weather_2, index=weather_df.index, columns=['Evaporation (mm)'])
# weather_2_df = weather_2_df.join(rain_df, how='right')
# weather_2_df_daily = weather_2_df.resample('D', how=np.sum)
# weather_2_df_daily = weather_2_df_daily.join(rain_df, how='left')
evaporation_2 = weather_641_df_daily['Evaporation (mm)']
# print evaporation_2
max_height_2 = 0.0
stage_volume_2 = default_stage_volume_file
stage_area_2 = default_stage_area_file
own_catchment_inflow_ratio_2 = 1.0
checkdam_2 = CheckdamParameters(check_dam_name=2, catchment_area=catchment_area_2, infiltration_rate=infiltration_rate, evaporation=evaporation_2, max_height=max_height_2, stage_volume_csv=stage_volume_2, stage_area_csv=stage_area_2,previous_check_dam=None, own_catchment_inflow_ratio=own_catchment_inflow_ratio_2)
checkdam_2.initial_volume = 0.0

had_chain_2 = CheckdamChain(inflow_catchment_area_df=inflow_catchment_area_had_df, check_dam_chain=[checkdam_639, checkdam_641, checkdam_2], slope=slope, intercept=intercept, output_df=had_chain_1.output_df)
# had_chain_2.create_output_df()
had_output_2_df = had_chain_2.simulate

# print had_output_2_df.head()

"""
# Merge had_chain_1 and had_chain_2 and simulate check dam 3
"""
# 3
catchment_area_3 = 0.244
# elevation_3 = 831
# ch_3_lat = 13.35314
# ch_3_long = 77.53556
# weather_3 = Open_Water_Evaporation(check_dam_name="2",air_temperature=airtemp, relative_humidity=hum, incoming_solar_radiation=rs, wind_speed_mps=wind_speed,elevation=elevation_3, date_time_index=weather_df.index, latitdude=ch_3_lat, longitude=ch_3_long)
# weather_3 = weather_3.calculate_half_hour_eo()
# weather_3_df = pd.DataFrame(weather_3, index=weather_df.index, columns=['Evaporation (mm)'])
# weather_3_df = weather_3_df.join(rain_df, how='right')
# weather_3_df_daily = weather_3_df.resample('D', how=np.sum)
# weather_3_df_daily = weather_3_df_daily.join(rain_df, how='left')
evaporation_3 = weather_641_df_daily['Evaporation (mm)']
# print evaporation_3
max_height_3 = 0.0
stage_volume_3 = default_stage_volume_file
stage_area_3 = default_stage_area_file
own_catchment_inflow_ratio_3 = 1.0
checkdam_3 = CheckdamParameters(check_dam_name=3, catchment_area=catchment_area_3, infiltration_rate=infiltration_rate, evaporation=evaporation_3, max_height=max_height_3, stage_volume_csv=stage_volume_3, stage_area_csv=stage_area_3,previous_check_dam=None, own_catchment_inflow_ratio=own_catchment_inflow_ratio_3)
checkdam_3.initial_volume = 0.0


had_chain_1_2 = CheckdamChainMerge(checkdam_chain_a=had_chain_1, checkdam_chain_b=had_chain_2, check_dam_a=1, check_dam_b=2, converging_check_dam=checkdam_3)
had_chain_1_2_df = had_chain_1_2.simulate

"""##################################

""" ####################################
"""
# Chain 3 625 -> 5
"""
# 625
catchment_area_625 = 0.35
elevation_625 = 830
ch_625_lat = 13.36411
ch_625_long = 77.55606
weather_625 = Open_Water_Evaporation(check_dam_name="625",air_temperature=airtemp, relative_humidity=hum, incoming_solar_radiation=rs, wind_speed_mps=wind_speed,elevation=elevation_625, date_time_index=weather_df.index, latitdude=ch_625_lat, longitude=ch_625_long)
weather_625 = weather_625.calculate_half_hour_eo()
weather_625_df = pd.DataFrame(weather_625, index=weather_df.index, columns=['Evaporation (mm)'])
# weather_625_df = weather_625_df.join(rain_df, how='right')
weather_625_df_daily = weather_625_df.resample('D', how=np.sum)
weather_625_df_daily = weather_625_df_daily.join(rain_df, how='left')
evaporation_625 = weather_625_df_daily['Evaporation (mm)']
# print evaporation_625
max_height_625 = 1.1
stage_volume_625 = default_stage_volume_file
stage_area_625 = default_stage_area_file
own_catchment_inflow_ratio_625 = 1.0
checkdam_625 = CheckdamParameters(check_dam_name=625, catchment_area=catchment_area_625, infiltration_rate=infiltration_rate, evaporation=evaporation_625, max_height=max_height_625, stage_volume_csv=stage_volume_625, stage_area_csv=stage_area_625,previous_check_dam=None, own_catchment_inflow_ratio=own_catchment_inflow_ratio_625)
checkdam_625.initial_volume = 0.0
checkdam_625.next_check_dam = 5

# 5
catchment_area_5 = 0.335
# elevation_5 = 830
# ch_5_lat = 13.36411
# ch_5_long = 77.55606
# weather_5 = Open_Water_Evaporation(check_dam_name="5",air_temperature=airtemp, relative_humidity=hum, incoming_solar_radiation=rs, wind_speed_mps=wind_speed,elevation=elevation_5, date_time_index=weather_df.index, latitdude=ch_5_lat, longitude=ch_5_long)
# weather_5 = weather_5.calculate_half_hour_eo()
# weather_5_df = pd.DataFrame(weather_5, index=weather_df.index, columns=['Evaporation (mm)'])
# weather_5_df = weather_5_df.join(rain_df, how='right')
# weather_5_df_daily = weather_5_df.resample('D', how=np.sum)
# weather_5_df_daily = weather_5_df_daily.join(rain_df, how='left')
evaporation_5 = weather_625_df_daily['Evaporation (mm)']
# print evaporation_5
max_height_5 = 0.0
stage_volume_5 = default_stage_volume_file
stage_area_5 = default_stage_area_file
own_catchment_inflow_ratio_5 = 1.0
checkdam_5 = CheckdamParameters(check_dam_name=5, catchment_area=catchment_area_5, infiltration_rate=infiltration_rate, evaporation=evaporation_5, max_height=max_height_5, stage_volume_csv=stage_volume_5, stage_area_csv=stage_area_5,previous_check_dam=checkdam_625, own_catchment_inflow_ratio=own_catchment_inflow_ratio_5)
checkdam_5.initial_volume = 0.0

had_chain_3 = CheckdamChain(inflow_catchment_area_df=inflow_catchment_area_had_df, check_dam_chain=[checkdam_625, checkdam_5], slope=slope, intercept=intercept)
had_output_3_df = had_chain_3.simulate

"""  ########################################

"""
Chain 4 627
"""
# 627
catchment_area_627 = 0.456
elevation_627 = 841
ch_627_lat = 13.36838
ch_627_long = 77.55807
weather_627 = Open_Water_Evaporation(check_dam_name="627",air_temperature=airtemp, relative_humidity=hum, incoming_solar_radiation=rs, wind_speed_mps=wind_speed,elevation=elevation_627, date_time_index=weather_df.index, latitdude=ch_627_lat, longitude=ch_627_long)
weather_627 = weather_627.calculate_half_hour_eo()
weather_627_df = pd.DataFrame(weather_627, index=weather_df.index, columns=['Evaporation (mm)'])
# weather_627_df = weather_627_df.join(rain_df, how='right')
weather_627_df_daily = weather_627_df.resample('D', how=np.sum)
weather_627_df_daily = weather_627_df_daily.join(rain_df, how='left')
evaporation_627 = weather_627_df_daily['Evaporation (mm)']
# print evaporation_627
max_height_627 = 1.2
stage_volume_627 = default_stage_volume_file
stage_area_627 = default_stage_area_file
own_catchment_inflow_ratio_627 = 1.0
checkdam_627 = CheckdamParameters(check_dam_name=627, catchment_area=catchment_area_627, infiltration_rate=infiltration_rate, evaporation=evaporation_627, max_height=max_height_627, stage_volume_csv=stage_volume_627, stage_area_csv=stage_area_627,previous_check_dam=None, own_catchment_inflow_ratio=own_catchment_inflow_ratio_627)
checkdam_627.initial_volume = 0.0

had_chain_4 = CheckdamChain(inflow_catchment_area_df=inflow_catchment_area_had_df, check_dam_chain=[checkdam_627], slope=slope, intercept=intercept)
had_output_4_df = had_chain_4.simulate

#test plots

"""
Chain 5 633
"""
# _633
catchment_area_633 = 0.145
elevation_633 = 839
ch_633_lat = 13.36838
ch_633_long = 77.55807
weather_633 = Open_Water_Evaporation(check_dam_name="633",air_temperature=airtemp, relative_humidity=hum, incoming_solar_radiation=rs, wind_speed_mps=wind_speed,elevation=elevation_633, date_time_index=weather_df.index, latitdude=ch_633_lat, longitude=ch_633_long)
weather_633 = weather_633.calculate_half_hour_eo()
weather_633_df = pd.DataFrame(weather_633, index=weather_df.index, columns=['Evaporation (mm)'])
# weather_633_df = weather_633_df.join(rain_df, how='right')
weather_633_df_daily = weather_633_df.resample('D', how=np.sum)
weather_633_df_daily = weather_633_df_daily.join(rain_df, how='left')
evaporation_633 = weather_633_df_daily['Evaporation (mm)']
# print evaporation_633
max_height_633 = 1.1
stage_volume_633 = default_stage_volume_file
stage_area_633 = default_stage_area_file
own_catchment_inflow_ratio_633 = 1.0
checkdam_633 = CheckdamParameters(check_dam_name=633, catchment_area=catchment_area_633, infiltration_rate=infiltration_rate, evaporation=evaporation_633, max_height=max_height_633, stage_volume_csv=stage_volume_633, stage_area_csv=stage_area_633,previous_check_dam=None, own_catchment_inflow_ratio=own_catchment_inflow_ratio_633)
checkdam_633.initial_volume = 0.0

had_chain_5 = CheckdamChain(inflow_catchment_area_df=inflow_catchment_area_had_df, check_dam_chain=[checkdam_633], slope=slope, intercept=intercept)
had_output_5_df = had_chain_5.simulate

"""
Merge Chain 4 and chain 5 and simulate for check dam 626
"""
# 626
catchment_area_626 = 0.051
elevation_626 = 834
ch_626_lat = 13.36586
ch_626_long = 77.5588
weather_626 = Open_Water_Evaporation(check_dam_name="626",air_temperature=airtemp, relative_humidity=hum, incoming_solar_radiation=rs, wind_speed_mps=wind_speed,elevation=elevation_626, date_time_index=weather_df.index, latitdude=ch_626_lat, longitude=ch_626_long)
weather_626 = weather_626.calculate_half_hour_eo()
weather_626_df = pd.DataFrame(weather_626, index=weather_df.index, columns=['Evaporation (mm)'])
# weather_626_df = weather_626_df.join(rain_df, how='right')
weather_626_df_daily = weather_626_df.resample('D', how=np.sum)
weather_626_df_daily = weather_626_df_daily.join(rain_df, how='left')
evaporation_626 = weather_626_df_daily['Evaporation (mm)']
# print evaporation_626
max_height_626 = 1.3
stage_volume_626 = default_stage_volume_file
stage_area_626 = default_stage_area_file
own_catchment_inflow_ratio_626 = 1.0
checkdam_626 = CheckdamParameters(check_dam_name=626, catchment_area=catchment_area_626, infiltration_rate=infiltration_rate, evaporation=evaporation_626, max_height=max_height_626, stage_volume_csv=stage_volume_626, stage_area_csv=stage_area_626,previous_check_dam=None, own_catchment_inflow_ratio=own_catchment_inflow_ratio_626)
checkdam_626.initial_volume = 0.0

had_chain_4_5 = CheckdamChainMerge(checkdam_chain_a=had_chain_4, checkdam_chain_b=had_chain_5, check_dam_a=627, check_dam_b=633, converging_check_dam=checkdam_626)
had_chain_4_5_df = had_chain_4_5.simulate

"""
Chain 6 634
"""
# 634
catchment_area_634 = 0.007
elevation_634 = 838
ch_634_lat = 13.36562
ch_634_long = 77.55905
weather_634 = Open_Water_Evaporation(check_dam_name="634",air_temperature=airtemp, relative_humidity=hum, incoming_solar_radiation=rs, wind_speed_mps=wind_speed,elevation=elevation_634, date_time_index=weather_df.index, latitdude=ch_634_lat, longitude=ch_634_long)
weather_634 = weather_634.calculate_half_hour_eo()
weather_634_df = pd.DataFrame(weather_634, index=weather_df.index, columns=['Evaporation (mm)'])
# weather_634_df = weather_634_df.join(rain_df, how='right')
weather_634_df_daily = weather_634_df.resample('D', how=np.sum)
weather_634_df_daily = weather_634_df_daily.join(rain_df, how='left')
evaporation_634 = weather_634_df_daily['Evaporation (mm)']
# print evaporation_634
max_height_634 = 0.61
stage_volume_634 = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_634/stage_vol.csv'
stage_area_634 = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_634/cont_area.csv'
own_catchment_inflow_ratio_634 = 1.0
checkdam_634 = CheckdamParameters(check_dam_name=634, catchment_area=catchment_area_634, infiltration_rate=infiltration_rate, evaporation=evaporation_634, max_height=max_height_634, stage_volume_csv=stage_volume_634, stage_area_csv=stage_area_634,previous_check_dam=None, own_catchment_inflow_ratio=own_catchment_inflow_ratio_634)
checkdam_634.initial_volume = 0.0

had_chain_6 = CheckdamChain(inflow_catchment_area_df=inflow_catchment_area_had_df, check_dam_chain=[checkdam_634], slope=slope, intercept=intercept)
had_output_6_df = had_chain_6.simulate

"""
Merge Chain 6 and chain 4_5  and simulate for check dam 624
"""
# 624
catchment_area_624 = 1
elevation_624 = 833
ch_624_lat = 13.3642
ch_624_long = 77.55827
weather_624 = Open_Water_Evaporation(check_dam_name="624",air_temperature=airtemp, relative_humidity=hum, incoming_solar_radiation=rs, wind_speed_mps=wind_speed,elevation=elevation_624, date_time_index=weather_df.index, latitdude=ch_624_lat, longitude=ch_624_long)
weather_624 = weather_624.calculate_half_hour_eo()
weather_624_df = pd.DataFrame(weather_624, index=weather_df.index, columns=['Evaporation (mm)'])
# weather_624_df = weather_624_df.join(rain_df, how='right')
weather_624_df_daily = weather_624_df.resample('D', how=np.sum)
weather_624_df_daily = weather_624_df_daily.join(rain_df, how='left')
evaporation_624 = weather_624_df_daily['Evaporation (mm)']
# print evaporation_624
max_height_624 = 1.3
stage_volume_624 = default_stage_volume_file
stage_area_624 = default_stage_area_file
own_catchment_inflow_ratio_624 = 1.0
checkdam_624 = CheckdamParameters(check_dam_name=624, catchment_area=catchment_area_624, infiltration_rate=infiltration_rate, evaporation=evaporation_624, max_height=max_height_624, stage_volume_csv=stage_volume_624, stage_area_csv=stage_area_624,previous_check_dam=None, own_catchment_inflow_ratio=own_catchment_inflow_ratio_624)
checkdam_624.initial_volume = 0.0
checkdam_624.next_check_dam = 6

had_chain_4_5_6 = CheckdamChainMerge(checkdam_chain_a=had_chain_4_5, checkdam_chain_b=had_chain_6, check_dam_a=626, check_dam_b=634, converging_check_dam=checkdam_624)
had_chain_4_5_6_df = had_chain_4_5_6.simulate

"""
Chain 7 had_chain_4_5_6 (624) -> 6
"""
# 6
catchment_area_6 = 0.263
# elevation_6 = 831
# ch_6_lat = 13.35314
# ch_6_long = 77.53556
# weather_6 = Open_Water_Evaporation(check_dam_name="6",air_temperature=airtemp, relative_humidity=hum, incoming_solar_radiation=rs, wind_speed_mps=wind_speed,elevation=elevation_6, date_time_index=weather_df.index, latitdude=ch_6_lat, longitude=ch_6_long)
# weather_6 = weather_6.calculate_half_hour_eo()
# weather_6_df = pd.DataFrame(weather_6, index=weather_df.index, columns=['Evaporation (mm)'])
# weather_6_df = weather_6_df.join(rain_df, how='right')
# weather_6_df_daily = weather_6_df.resample('D', how=np.sum)
# weather_6_df_daily = weather_6_df_daily.join(rain_df, how='left')
evaporation_6 = weather_624_df_daily['Evaporation (mm)']
# print evaporation_6
max_height_6 = 0.0
stage_volume_6 = default_stage_volume_file
stage_area_6 = default_stage_area_file
own_catchment_inflow_ratio_6 = 1.0
checkdam_6 = CheckdamParameters(check_dam_name=6, catchment_area=catchment_area_6, infiltration_rate=infiltration_rate, evaporation=evaporation_6, max_height=max_height_6, stage_volume_csv=stage_volume_6, stage_area_csv=stage_area_6, previous_check_dam=checkdam_624, own_catchment_inflow_ratio=own_catchment_inflow_ratio_6)
checkdam_6.initial_volume = 0.0

had_chain_7 = CheckdamChain(inflow_catchment_area_df=inflow_catchment_area_had_df, check_dam_chain=[checkdam_6], slope=slope, intercept=intercept, output_df=had_chain_4_5_6.output_df)
# had_chain_2.create_output_df()
had_output_7_df = had_chain_7.simulate

for checkdam in had_chain_4_5_6.checkdam_list:
#     print checkdam.check_dam_name, checkdam.max_volume
    fig = plt.figure()
    plt.plot(had_chain_4_5_6.output_df.index, had_chain_4_5_6.output_df[('total_inflow_{0:d}'.format(checkdam.check_dam_name))], '-', label=('inflow_{0:d}'.format(checkdam.check_dam_name)))
    plt.plot(had_chain_4_5_6.output_df.index, had_chain_4_5_6.output_df[('evap_{0:d}'.format(checkdam.check_dam_name))], '-', label=('evap_{0:d}'.format(checkdam.check_dam_name)))
    plt.plot(had_chain_4_5_6.output_df.index, had_chain_4_5_6.output_df[('volume_{0:d}'.format(checkdam.check_dam_name))], '-', label=('volume_{0:d}'.format(checkdam.check_dam_name)))
    plt.plot(had_chain_4_5_6.output_df.index, had_chain_4_5_6.output_df[('infilt_{0:d}').format(checkdam.check_dam_name)], '-', label=('infilt_{0:d}'.format(checkdam.check_dam_name)))
    plt.plot(had_chain_4_5_6.output_df.index, had_chain_4_5_6.output_df[('overflow_{0:d}'.format(checkdam.check_dam_name))], '-', label=('overflow_{0:d}'.format(checkdam.check_dam_name)))
    plt.hlines(y=checkdam.max_volume, xmin=(min(had_chain_4_5_6.output_df.index)), xmax=max(had_chain_4_5_6.output_df.index), linewidth=2, color='k')
    plt.legend()
# plt.show()

for checkdam in had_chain_7.check_dam_chain:
    print checkdam.check_dam_name, checkdam.max_volume
    fig = plt.figure()
    plt.plot(had_chain_7.output_df.index, had_chain_7.output_df[('total_inflow_{0:d}'.format(checkdam.check_dam_name))], '-', label=('inflow_{0:d}'.format(checkdam.check_dam_name)))
    plt.plot(had_chain_7.output_df.index, had_chain_7.output_df[('evap_{0:d}'.format(checkdam.check_dam_name))], '-', label=('evap_{0:d}'.format(checkdam.check_dam_name)))
    plt.plot(had_chain_7.output_df.index, had_chain_7.output_df[('volume_{0:d}'.format(checkdam.check_dam_name))], '-', label=('volume_{0:d}'.format(checkdam.check_dam_name)))
    plt.plot(had_chain_7.output_df.index, had_chain_7.output_df[('infilt_{0:d}').format(checkdam.check_dam_name)], '-', label=('infilt_{0:d}'.format(checkdam.check_dam_name)))
    plt.plot(had_chain_7.output_df.index, had_chain_7.output_df[('overflow_{0:d}'.format(checkdam.check_dam_name))], '-', label=('overflow_{0:d}'.format(checkdam.check_dam_name)))
    plt.hlines(y=checkdam.max_volume, xmin=(min(had_chain_7.output_df.index)), xmax=max(had_chain_7.output_df.index), linewidth=2, color='k')
    plt.legend()
plt.show()


# for checkdam in had_chain_2.check_dam_chain:
#     print checkdam.check_dam_name, checkdam.max_volume
#     fig = plt.figure()
#     plt.plot(had_chain_2.output_df.index, had_chain_2.output_df[('total_inflow_{0:d}'.format(checkdam.check_dam_name))], '-', label=('inflow_{0:d}'.format(checkdam.check_dam_name)))
#     plt.plot(had_chain_2.output_df.index, had_chain_2.output_df[('evap_{0:d}'.format(checkdam.check_dam_name))], '-', label=('evap_{0:d}'.format(checkdam.check_dam_name)))
#     plt.plot(had_chain_2.output_df.index, had_chain_2.output_df[('volume_{0:d}'.format(checkdam.check_dam_name))], '-', label=('volume_{0:d}'.format(checkdam.check_dam_name)))
#     plt.plot(had_chain_2.output_df.index, had_chain_2.output_df[('infilt_{0:d}').format(checkdam.check_dam_name)], '-', label=('infilt_{0:d}'.format(checkdam.check_dam_name)))
#     plt.plot(had_chain_2.output_df.index, had_chain_2.output_df[('overflow_{0:d}'.format(checkdam.check_dam_name))], '-', label=('overflow_{0:d}'.format(checkdam.check_dam_name)))
#     plt.hlines(y=checkdam.max_volume, xmin=(min(had_chain_2.output_df.index)), xmax=max(had_chain_2.output_df.index), linewidth=2, color='k')
#     plt.legend()
# plt.show()
# had_output_1_df.to_csv('/media/kiruba/New Volume/milli_watershed/cumulative impacts/had_chain_1.csv')

# for checkdam in had_chain_4_5_6.checkdam_list:
# #     print checkdam.check_dam_name, checkdam.max_volume
#     fig = plt.figure()
#     plt.plot(had_chain_4_5.output_df.index, had_chain_4_5.output_df[('total_inflow_{0:d}'.format(checkdam.check_dam_name))], '-', label=('inflow_{0:d}'.format(checkdam.check_dam_name)))
#     plt.plot(had_chain_4_5.output_df.index, had_chain_4_5.output_df[('evap_{0:d}'.format(checkdam.check_dam_name))], '-', label=('evap_{0:d}'.format(checkdam.check_dam_name)))
#     plt.plot(had_chain_4_5.output_df.index, had_chain_4_5.output_df[('volume_{0:d}'.format(checkdam.check_dam_name))], '-', label=('volume_{0:d}'.format(checkdam.check_dam_name)))
#     plt.plot(had_chain_4_5.output_df.index, had_chain_4_5.output_df[('infilt_{0:d}').format(checkdam.check_dam_name)], '-', label=('infilt_{0:d}'.format(checkdam.check_dam_name)))
#     plt.plot(had_chain_4_5.output_df.index, had_chain_4_5.output_df[('overflow_{0:d}'.format(checkdam.check_dam_name))], '-', label=('overflow_{0:d}'.format(checkdam.check_dam_name)))
#     plt.hlines(y=checkdam.max_volume, xmin=(min(had_chain_4_5.output_df.index)), xmax=max(had_chain_4_5.output_df.index), linewidth=2, color='k')
#     plt.legend()
# plt.show()