__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from fractions import Fraction
from bisect import bisect_left
import math
from datetime import datetime
from datetime import timedelta
import meteolib as met
# import evaplib
import scipy as sp
import mynormalize

date_format = '%Y-%m-%d %H:%M:%S'
daily_format = '%Y-%m-%d'

# check dam module


def spread(start, end, count, mode=1):
    """

    Yield a sequence of evenly-spaced numbers between start and end.

    spread(start, end, count [, mode]) -> generator

    The range start...end is divided into count evenly-spaced (or as close to
    evenly-spaced as possible) intervals. The end-points of each interval are
    then yielded, optionally including or excluding start and end themselves.
    By default, start is included and end is excluded.

    Optional argument mode controls whether spread() includes the start and
    end values. mode must be an int. Bit zero of mode controls whether start
    is included (on) or excluded (off); bit one does the same for end. Hence:

    0 -> open interval (start and end both excluded)
    1 -> half-open (start included, end excluded)
    2 -> half open (start excluded, end included)
    3 -> closed (start and end both included)

    By default, mode=1 and only start is included in the output.

    (Note: depending on mode, the number of values returned can be count, count-1 or count+1.)

    :param start: starting number
    :param end: ending number
    :param count: number of values returned
    :param mode: controls the output, default is 1
    :return: generator

    Examples:
        >>> list(spread(0.0, 2.1, 3))
        [0.0, 0.7, 1.4]
    """
    if not isinstance(mode, int):
        raise TypeError('mode must be an int')
    if count != int(count):
        raise ValueError('count must be an integer')
    if count <= 0:
        raise ValueError('count must be positive')
    if mode & 1:
        yield start
    width = Fraction(end - start)
    start = Fraction(start)
    for i in range(1, count):
        yield float(start + i * width / count)
    if mode & 2:
        yield end


def pairwise(iterable):
    """
    s -> (s0,s1), (s1,s2), (s2,s3)

    :param iterable:
    :return:
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


def calcvolume(y_value_list, elevation_data, dam_height):
    """
    Modified function to calculate stage vs volume relationship from elevation data

    :param y_value_list: List of Y values, y1, y2,...
    :param elevation_data: Elevation data with headers df.Yy1, df.Yy2
    :param dam_height: check dam height in metre
    :return: pandas dataframe with stage and corresponding volume
    """
    no_of_stage_interval = dam_height / 0.05
    dz = list((spread(0.00, dam_height, int(no_of_stage_interval), mode=3)))
    index = [range(len(dz))]  # no of stage intervals
    columns = ['stage_m']
    data = np.array(dz)
    output = pd.DataFrame(data, index=index, columns=columns)
    for l1, l2 in pairwise(y_value_list):
        results = []
        profile = elevation_data["Y_%s" % float(l1)]
        order = l1
        dy = int(l2 - l1)
        for stage in dz:
            water_area = 0
            for z1, z2 in pairwise(profile):
                delev = (z2 - z1) / 10
                elev = z1
                for b in range(1, 11, 1):
                    elev += delev
                    if stage > elev:
                        water_area += (0.1 * (stage - elev))
                calc_vol = water_area * dy
            results.append(calc_vol)

        output[('Volume_%s' % order)] = results

    output_series = output.filter(regex="Volume_")
    output["total_vol_cu_m"] = output_series.sum(axis=1)
    final_results = output[['stage_m', "total_vol_cu_m"]]
    return final_results


def find_range(array, x):
    """
    Function to calculate bounding intervals from array to do piecewise linear interpolation.

    :param array: list of values
    :param x: interpolation value
    :return: boundary interval

    Examples:
        >>> array = [0, 1, 2, 3, 4]
        >>>find_range(array, 1.5)
        1, 2
    """
    if x < max(array):
        start = bisect_left(array, x)
        return array[start-1], array[start]
    else:
        return min(array), max(array)


def fill_profile(base_df, slope_df, midpoint_index):
    """
    Function to fill profile data where only slope data is collected.
    The difference between two slope is added to previous slope to get the current
    cross section.

    :param base_df:  base profile
    :param slope_df: slope profile
    :param midpoint_index: index of midpoint(x=0)
    :return: filled base profile
    """
    base_z = base_df.ix[midpoint_index, 0:]
    slope_z = slope_df.ix[:, 1]
    base_y = base_z.index
    slope_y = slope_df.ix[:, 0]
    slope_z.index = slope_y
    new_base_df = base_df
    for y_s in slope_z.index:
        if y_s not in base_z.index.tolist():
            y_t = find_range(base_y, y_s)
            template = base_df[y_t]
            z1 = template.ix[midpoint_index, ]
            z2 = slope_z[y_s]
            diff = z2 - z1
            profile = template + diff
            profile.name = y_s
            new_base_df = new_base_df.join(profile, how='right')
    return new_base_df


def set_column_sequence(dataframe, seq):
    """Takes a dataframe and a sequence of its columns, returns dataframe with seq as first column"""
    cols = seq[:]  # copy so we don't mutate seq
    for x in dataframe.columns:
        if x not in cols:
            cols.append(x)
    return dataframe[cols]


def contour_area(mpl_obj):
    """
    Returns a array of contour levels and
    corresponding cumulative area of contours
    # Refer: Nikolai Shokhirev http://www.numericalexpert.com/blog/area_calculation/

    :param mpl_obj: Matplotlib contour object
    :return: [(level1, area1), (level1, area1+area2)]
    """
    global poly_area
    n_c = len(mpl_obj.collections)  # n_c = no of contours
    print 'No. of contours = {0}'.format(n_c)
    area = 0.0000
    cont_area_array = []
    for contour in range(n_c):
        n_p = len(mpl_obj.collections[contour].get_paths())
        zc = mpl_obj.levels[contour + 1]
        for path in range(n_p):
            p = mpl_obj.collections[contour].get_paths()[path]
            v = p.vertices
            l = len(v)
            s = 0.0000
            for i in range(l):
                j = (i + 1) % l
                s += (v[j, 0] - v[i, 0]) * (v[j, 1] + v[i, 1])
                poly_area = 0.5 * abs(s)
            area += poly_area
        cont_area_array.append((zc, area))
    return cont_area_array


def polyfit(x, y, degree):
    """
    Wrapper around np.polyfit

    :param x: x values
    :param y: y values
    :param degree: polynomial degree
    :return: results, polynomial has coefficients, determination has r-squared
    :rtype: dict
    """
    results = {}
    coeffs = np.polyfit(x, y, degree)
    results['polynomial'] = coeffs.tolist()
    # r squared
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    results['determination'] = ssreg / sstot
    return results


def myround(a, decimals=1):
    """
    Function to round, better than numpy around

    :param a: float to be rounded
    :param decimals: no of decimal places, default = 1
    :return: float

    Examples:
        >>> myround(0.7568,decimals=2)
        0.76
    """
    return np.around(a - 10 ** (-(decimals + 5)), decimals=decimals)


def read_correct_ch_dam_data(csv_file, calibration_slope, calibration_intercept, stage_cutoff=0.1):
    """
    Function to read and calibrate odyssey capacitance sensor data

    :param csv_file: csv file created from sensor
    :param calibration_slope: slope
    :param calibration_intercept: intercept
    :return: calibrated and time corrected data

    Examples:
        >>> read_correct_ch_dam_data(csv_file=file.csv, calibration_slope=0.111, calibration_intercept=0.222)
    """
    water_level = pd.read_csv(csv_file, skiprows=9, sep=',', header=0,
                              names=['scan no', 'date', 'time', 'raw value', 'calibrated value'])
    water_level['calibrated value'] = (water_level['raw value'] * calibration_slope) + calibration_intercept  # in cm
    # water_level['calibrated value'] = np.round(water_level['calibrated value']/resolution_ody)*resolution_ody
    water_level['calibrated value'] /= 1000.0
    water_level['calibrated value'] = myround(a=water_level['calibrated value'], decimals=3)
    # #change the column name
    water_level.columns.values[4] = 'stage(m)'
    # print water_level.head()

    # create date time index
    format = '%d/%m/%Y  %H:%M:%S'
    c_str = ' 24:00:00'
    for index, row in water_level.iterrows():
        x_str = row['time']
        if x_str == c_str:
            # convert string to datetime object
            r_date = pd.to_datetime(row['date'], format='%d/%m/%Y ')
            # add 1 day
            c_date = r_date + timedelta(days=1)
            # convert datetime to string
            c_date = c_date.strftime('%d/%m/%Y ')
            c_time = ' 00:00:00'
            # water_level.loc[:, ('date', index)] = c_date
            # water_level.loc[:, ('time', index)] = c_time
            water_level.loc[index,'date'] = c_date
            water_level.loc[index,'time'] = c_time

    water_level['date_time'] = pd.to_datetime(water_level['date'] + water_level['time'], format=format)
    water_level.set_index(water_level['date_time'], inplace=True)
    # # drop unneccessary columns before datetime aggregation
    for index, row in water_level.iterrows():
        # print row
        obs_stage = row['stage(m)']
        if obs_stage < stage_cutoff:
            # water_level.loc[:, ('stage(m)', index.strftime(date_format))] = 0.0
            water_level.loc[index,'stage(m)'] = 0.0

    water_level.drop(['scan no', 'date', 'time', 'date_time'], inplace=True, axis=1)

    return water_level


def extraterrestrial_irrad(local_datetime, latitude_deg, longitude_deg):
    """
    Calculates extraterrestrial radiation in MJ/m2/30min

    :param local_datetime: datetime object
    :param latitude_deg: in decimal degree
    :param longitude_deg: in decimal degree
    :return: Extra terrestrial radiation in MJ/m2/30min
    :rtype: float
    """

    s = 0.0820  # MJ m-2 min-1
    lat_rad = latitude_deg * (math.pi / 180)
    day = (local_datetime - datetime(local_datetime.year, 1, 1)).days + 1
    hour = float(local_datetime.hour)
    minute = float(local_datetime.minute)
    b = ((2 * math.pi) * (day - 81)) / 364
    sc = 0.1645 * (math.sin(2 * b)) - 0.1255 * (math.cos(b)) - 0.025 * (math.sin(b))  # seasonal correction in hour
    lz = 270  # for India longitude of local time zone in degrees west of greenwich
    lm = (180 + (180 - longitude_deg))  # longitude of measurement site
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
        rext = ((12 * 60) / math.pi) * s * dr * (((w2 - w1) * math.sin(lat_rad) * math.sin(dt)) + (
            math.cos(lat_rad) * math.cos(dt) * (math.sin(w2) - math.sin(w1))))  # MJm-2(30min)-1
    return rext


"""
Open water evaporation function for half hour
Modified from evaplib.py
http://python.hydrology-amsterdam.nl/moduledoc/index.html#module-evaplib
"""


def delta_calc(airtemp):
    """
    Calculates slope of saturation vapour pressure curve at air temperature [kPa/Celsius]
    http://www.fao.org/docrep/x0490e/x0490e07.htm

    :param airtemp: Temperature in Celsius
    :return: slope of saturation vapour pressure curve [kPa/Celsius]
    """
    l = sp.size(airtemp)
    if l < 2:
        temp = airtemp + 237.3
        b = 0.6108 * (math.exp((17.27 * airtemp) / temp))
        delta = (4098 * b) / (temp ** 2)
    else:
        delta = sp.zeros(l)
        for i in range(0, l):
            temp = airtemp[i] + 237.3
            b = 0.6108 * (math.exp(17.27 * airtemp[i]) / temp)
            delta[i] = (4098 * b) / (temp ** 2)
    return delta


def half_hour_evaporation(airtemp=sp.array([]),
                          rh=sp.array([]),
                          airpress=sp.array([]),
                          rs=sp.array([]),
                          rext=sp.array([]),
                          u=sp.array([]),
                          z=0.0):
    """
    Function to calculate daily Penman open water evaporation (in mm/30min).
    Equation according to
    Shuttleworth, W. J. 2007. "Putting the 'Vap' into Evaporation."
    Hydrology and Earth System Sciences 11 (1): 210-44. doi:10.5194/hess-11-210-2007.

    :param airtemp: average air temperature [Celsius]
    :param rh: relative humidity[%]
    :param airpress: average air pressure[Pa]
    :param rs: Incoming solar radiation [MJ/m2/30min]
    :param rext: Extraterrestrial radiation [MJ/m2/30min]
    :param u: average wind speed at 2 m from ground [m/s]
    :param z: site elevation, default is zero [metre]
    :return: Penman open water evaporation values [mm/30min]

    Examples:
          >>> half_hour_evaporation = E0(T,RH,press,Rs,N,Rext,u,1000.0) # at 1000 m a.s.l
    """
    # Set constants
    albedo = 0.06  # open water albedo
    # Stefan boltzmann constant = 5.670373*10-8 J/m2/k4/s
    # http://en.wikipedia.org/wiki/Stefan-Boltzmann_constant
    # sigma = 5.670373*(10**-8)  # J/m2/K4/s
    sigma = (1.02066714 * (10 ** -10))  # Stefan Boltzmann constant MJ/m2/K4/30min
    # Calculate Delta, gamma and lambda
    delta = delta_calc(airtemp)  # [Kpa/C]
    # Lambda = met.L_calc(airtemp)/(10**6) # [MJ/Kg]
    # gamma = met.gamma_calc(airtemp, rh, airpress)/1000
    # Lambda = 2.501 -(0.002361*airtemp)     # [MJ/kg]
    # gamma = (0.0016286 *(airpress/1000))/Lambda
    # Calculate saturated and actual water vapour pressure
    es = met.es_calc(airtemp)  # [Pa]
    ea = met.ea_calc(airtemp, rh)  # [Pa]
    # Determine length of array
    l = sp.size(airtemp)
    # Check if we have a single value or an array
    if l < 2:
        lambda_mj_kg = 2.501 - (0.002361 * airtemp)  # [MJ/kg]
        gamma = (0.0016286 * (airpress / 1000)) / lambda_mj_kg
        rns = (1.0 - albedo) * rs  # shortwave component [MJ/m2/30min]
        # calculate clear sky radiation Rs0
        rs0 = (0.75 + (2E-5 * z)) * rext
        f = (1.35 * (rs / rs0)) - 0.35
        epsilom = 0.34 - (-0.14 * sp.sqrt(ea / 1000))
        rnl = f * epsilom * sigma * (airtemp + 273.16) ** 4  # Longwave component [MJ/m2/30min]
        rnet = rns - rnl
        Ea = (1 + (0.536 * u)) * ((es / 1000) - (ea / 1000))
        E0 = ((delta * rnet) + gamma * (6.43 * Ea)) / (lambda_mj_kg * (delta + gamma))
    else:
        # Inititate output array
        E0 = sp.zeros(l)
        rns = sp.zeros(l)
        rs0 = sp.zeros(l)
        f = sp.zeros(l)
        epsilom = sp.zeros(l)
        rnl = sp.zeros(l)
        rnet = sp.zeros(l)
        Ea = sp.zeros(l)
        lambda_mj_kg = sp.zeros(l)
        gamma = sp.zeros(l)
        for i in range(0, l):
            lambda_mj_kg[i] = 2.501 - (0.002361 * airtemp[i])
            gamma[i] = (0.0016286 * (airpress[i] / 1000)) / lambda_mj_kg[i]
            # calculate longwave radiation (MJ/m2/30min)
            rns[i] = (1.0 - albedo) * rs[i]
            # calculate clear sky radiation Rs0
            rs0[i] = (0.75 + (2E-5 * z))
            f[i] = (1.35 * (rs[i] / rs0[i])) - 0.35
            epsilom[i] = 0.34 - (-0.14 * sp.sqrt(ea[i] / 1000))
            rnl[i] = f[i] * epsilom[i] * sigma * (airtemp[i] + 273.16) ** 4  # Longwave component [MJ/m2/30min]
            rnet[i] = rns[i] - rnl[i]
            Ea[i] = (1 + (0.536 * u[i])) * ((es[i] / 1000) - (ea[i] / 1000))
            E0[i] = ((delta[i] * rnet[i]) + gamma[i] * (6.43 * Ea[i])) / (lambda_mj_kg[i] * (delta[i] + gamma[i]))
    return E0


def pick_incorrect_value(dataframe, **param):
    """
    Selects a unique list of timestamp that satisfies the condition given in the param dictionary

    :param dataframe: Pandas dataframe
    :param param: Conditonal Dictionary, Eg.{column name: [cutoff, '>']}
    :type param: dict
    :return: unique list of timestamp
    :rtype: list
    """
    wrong_date_time = []
    unique_list = []
    # first_time = pd.to_datetime('2014-05-15 18:00:00', format='%Y-%m-%d %H:%M:%S')
    # final_time = pd.to_datetime('2014-09-09 23:00:00', format='%Y-%m-%d %H:%M:%S')
    for key, value in param.items():
        # print key
        # print len(wrong_date_time)
        if value[1] == '>':
            wrong_df = dataframe[dataframe[key] > value[0]]
        if value[1] == '<':
            wrong_df = dataframe[dataframe[key] < value[0]]
        if value[1] ==  '=':
            wrong_df = dataframe[dataframe[key] == value[0]]
        for wrong_time in wrong_df.index:
            if max(dataframe.index) > wrong_time > min(dataframe.index):
                wrong_date_time.append(wrong_time)
            # if final_time > wrong_time > first_time:


    for i in wrong_date_time:
        if i not in unique_list:
            unique_list.append(i)

    return unique_list


def day_interpolate(dataframe, column_name, wrong_date_time):
    """
    Function to do linear interpolate on daily timescale

    :param dataframe: Pandas dataframe
    :param column_name: Interpolation target column name of dataframe
    :type column_name: str
    :param wrong_date_time: List of error timestamp
    :type wrong_date_time: list
    :return: Corrected dataframe
    """
    initial_cutoff = min(dataframe.index) + timedelta(days=1)
    final_cutoff = max(dataframe.index) - timedelta(days=1)
    for date_time in wrong_date_time:
        if (date_time > initial_cutoff ) and (date_time < final_cutoff):
            prev_date_time = date_time - timedelta(days=1)
            next_date_time = date_time + timedelta(days=1)
            prev_value = dataframe[column_name][prev_date_time.strftime('%Y-%m-%d %H:%M')]
            next_value = dataframe[column_name][next_date_time.strftime('%Y-%m-%d %H:%M')]
            average_value = 0.5*(prev_value + next_value)
            dataframe[column_name][date_time.strftime('%Y-%m-%d %H:%M')] = average_value

    return dataframe

def previous_interpolate(dataframe, column_name, wrong_date_time):
    """

    Function to fill the missing values from corresponding previous day's time period

    :param dataframe: Pandas dataframe
    :param column_name: Interpolation target column name of dataframe
    :type column_name: str
    :param wrong_date_time: List of error timestamp
    :type wrong_date_time: list
    :return: Corrected dataframe
    """
    initial_cutoff = min(dataframe.index) + timedelta(days=1)
    final_cutoff = max(dataframe.index) - timedelta(days=1)
    for date_time in wrong_date_time:
        if (date_time > initial_cutoff ) and (date_time < final_cutoff):
            prev_date_time = date_time - timedelta(days=1)
            prev_value = dataframe[column_name][prev_date_time.strftime('%Y-%m-%d %H:%M')]
            dataframe[column_name][date_time.strftime('%Y-%m-%d %H:%M')] = prev_value

    return dataframe


class DraggableColorbar(object):
    def __init__(self, cbar, mappable):
        self.cbar = cbar
        self.mappable = mappable
        self.press = None
        self.cycle = sorted([i for i in dir(plt.cm) if hasattr(getattr(plt.cm,i),'N')])
        self.index = self.cycle.index(cbar.get_cmap().name)

    def connect(self):
        """connect to all the events we need"""
        self.cidpress = self.cbar.patch.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.cbar.patch.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.cbar.patch.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.keypress = self.cbar.patch.figure.canvas.mpl_connect(
            'key_press_event', self.key_press)

    def on_press(self, event):
        """on button press we will see if the mouse is over us and store some data"""
        if event.inaxes != self.cbar.ax: return
        self.press = event.x, event.y

    def key_press(self, event):
        if event.key=='down':
            self.index += 1
        elif event.key=='up':
            self.index -= 1
        if self.index<0:
            self.index = len(self.cycle)
        elif self.index>=len(self.cycle):
            self.index = 0
        cmap = self.cycle[self.index]
        self.cbar.set_cmap(cmap)
        self.cbar.draw_all()
        self.mappable.set_cmap(cmap)
        self.mappable.get_axes().set_title(cmap)
        self.cbar.patch.figure.canvas.draw()

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.cbar.ax: return
        xprev, yprev = self.press
        dx = event.x - xprev
        dy = event.y - yprev
        self.press = event.x,event.y
        #print 'x0=%f, xpress=%f, event.xdata=%f, dx=%f, x0+dx=%f'%(x0, xpress, event.xdata, dx, x0+dx)
        scale = self.cbar.norm.vmax - self.cbar.norm.vmin
        perc = 0.03
        if event.button==1:
            self.cbar.norm.vmin -= (perc*scale)*np.sign(dy)
            self.cbar.norm.vmax -= (perc*scale)*np.sign(dy)
        elif event.button==3:
            self.cbar.norm.vmin -= (perc*scale)*np.sign(dy)
            self.cbar.norm.vmax += (perc*scale)*np.sign(dy)
        self.cbar.draw_all()
        self.mappable.set_norm(self.cbar.norm)
        self.cbar.patch.figure.canvas.draw()


    def on_release(self, event):
        """on release we reset the press data"""
        self.press = None
        self.mappable.set_norm(self.cbar.norm)
        self.cbar.patch.figure.canvas.draw()

    def disconnect(self):
        """disconnect all the stored connection ids"""
        self.cbar.patch.figure.canvas.mpl_disconnect(self.cidpress)
        self.cbar.patch.figure.canvas.mpl_disconnect(self.cidrelease)
        self.cbar.patch.figure.canvas.mpl_disconnect(self.cidmotion)
