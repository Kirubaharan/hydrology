# -*- coding: utf-8 -*-
""" Meteolib: A libray with Python functions for calculations of micrometeorological parameters.
    
    Miscellaneous functions:
        - dataload: Loads an ASCII data file into an array
        - event2time: Convert (event) based measurements into equidistant time spaced data for a selected interval
        - date2doy: Calculates day of year from day, month and year data
        - sun_NR: Maximum sunshine duration [h] and extraterrestrial radiation [J/day]

    Meteorological functions: 
        - es_calc:    Calculate saturation vapour pressures
        - ea_calc:    Calculate actual vapour pressures
        - vpd_calc:   Calculate vapour pressure deficits
        - Delta_calc: Calculate slope of vapour pressure curve
        - L_calc:     Calculate latent heat of vapourisation
        - cp_calc:    Calculate specific heat 
        - gamma_calc: Calculate psychrometric constant
        - rho_calc:   Calculate air density
        - pottemp:    Calculate potential temperature (1000 hPa reference pressure)
        - windvec:    Calculate average wind direction and speed
    
"""

__author__ = "Maarten J. Waterloo <m.j.waterloo@vu.nl>"
__version__ = "1.0"
__date__ = "Nov 2012"

    
# Load some relevant python functions
import math     # import math library
import scipy    # import scientific python functions
import datetime # Get date and time module

'''
    ================================================================
    General functions for loading and saving data in files
    ================================================================
'''

def dataload(datafile, delimiter):
    '''
    Function to load data from an ASCII text file "filename" with values being
    separated by a "delimiter".

    Usage:
        data = dataload(filename,delimiter)
    
    This will place the data from the file into an array "data" that can be used
    for further calculations. Both filename and delimiter are of type string.

    The data must be regular, i.e. the same number of values should appear in
    every row.

    Examples
        >>> data = dataload('meteo.dat',',')  # meteo.dat comma delimited
        >>> data = dataload('meteo.dat',' ')  # meteo.dat space delimited
        >>> # Now using variables
        >>> filedir = 'd:/workdir/meteo.dat'
        >>> separator = '\t'                  # tab delimited data file
        >>>
        >>> data = loadtxt(filedir,separator) 
    '''
    
    # Open and read the file
    inputfile = file(datafile, "r");
    
    # Define array which will hold all data
    alldata = []
    
    # Start reading the data from the file into the linedata array
    for line in inputfile.readlines():
        # Define array holding single line of data
        linedata = []
        # split the dataline at "separator" (tab = \t) and get the values
        for value in line.split(delimiter):
            # Append each value in a line so that we have a complete data
            # line 
            linedata.append(float(value))
        # Append the values in linedata to the alldata array    
        alldata.append(linedata)
    # Now close the file
    inputfile.close()
    
    # Store the data in an array
    data_array = scipy.qarray(alldata)
    
    # Make the data array available as function output 
    return(data_array)


def event2time(yyyy=scipy.array([]), doytime=scipy.array([]), \
               X=scipy.array([]), method=str, interval=None):
    '''
    Function to convert (event-based) time series data to equidistant time spaced data (Nov. 2012) at a specified interval.
    The maximum interval for processing is 86400 s, resulting in daily values.
    You can choose to sum (e.g. for event-based rainfall measurements) or average the input data over a given time interval.
    If you choose to average, a -9999 value (missing value) will be output if there are no data in the specified interval. For summation,
    a zero will be output, as required for event-based rainfall measurements.
    
    
    Input:
        - yyyy: Array of year values (e.g. 2008)
        - doytime: Array of day of year (doy, 1-366) + decimal time values (0-1) (e.g. 133.4375)
        - X: Array of data values (e.g. 0.2). for event-based precipitation data, data should be zero at start and end times of the event data record
        - method: Enter 'sum' to sum data (e.g. precipitation), and 'avg' to average data over the selected time interval
        - interval: Optional: interval in seconds (integer value, e.g. 900 for a 15-minute interval). A default value of 1800 s is assumed when interval is not specified as a function argument
    
    Output:
        - YEAR: Array of year
        - DOY_TIME: Array of day of year (1-366) + decimal time values (0-1), e.g. 153.5 for noon on day 153.
        - Y: Array of corresponding summed or averaged data values, where -9999 indicates missing values when 'avg' is selected and 0 when 'sum' is se;ected.
    
    Examples:
        >>> import meteolib
        >>> year=[2008,2008,2008,2008,2008]
        >>> daytime=[153.5,153.9,154.1,154.3,154.4]
        >>> vals=[0,0.4,2.3,2.9,0]
        >>> meteolib.event2time(year,daytime,vals,'sum',3600)
        (array([ 2008, 2008, 2008, 2008, 2008, 2008, 2008, 2008, 2008, 2008, 2008,
        2008, 2008, 2008, 2008, 2008, 2008, 2008, 2008, 2008]),
        array([ 153.58333333, 153.625, 153.66666667, 153.70833333, 153.75,
        153.79166667, 153.83333333, 153.875, 153.91666667, 153.95833333, 154.,
        154.04166667, 154.08333333, 154.125, 154.16666667, 154.20833333, 154.25,
        154.29166667, 154.33333333, 154.375 ]),
        array([ 0.4, 0., 0., 0., 0., 0., 0., 0., 2.3, 0., 0., 0., 0., 2.9, 0.,
        0., 0., 0., 0., 0. ]))
        >>> event2time(year,daytime,vals,'avg',3600)
        (array([ 2008, 2008, 2008, 2008, 2008, 2008, 2008, 2008, 2008, 2008, 2008,
        2008, 2008, 2008, 2008, 2008, 2008, 2008, 2008, 2008 ]),
        array([ 153.58333333, 153.625, 153.66666667, 153.70833333, 153.75,
        153.79166667, 153.83333333, 153.875, 153.91666667, 153.95833333, 154.,
        154.04166667, 154.08333333, 154.125, 154.16666667, 154.20833333, 154.25,
        154.29166667, 154.33333333, 154.375 ]),
        array([ 4.00000000e-01, -9.99900000e+03, -9.99900000e+03, -9.99900000e+03,
        -9.99900000e+03, -9.99900000e+03, -9.99900000e+03, -9.99900000e+03,
        2.30000000e+00, -9.99900000e+03, -9.99900000e+03, -9.99900000e+03,
        -9.99900000e+03, 2.90000000e+00, -9.99900000e+03, -9.99900000e+03,
        -9.99900000e+03, -9.99900000e+03, 0.00000000e+00, -9.99900000e+03 ]))
        >>> yr,day_time,sum_P = event2time(year,daytime,vals,'sum',3600)
    '''

    # Check for correct method input
    if method != 'sum':
        if method != 'avg':
            print('WARNING: method input unknown - set to default \'sum\'! \n')
            method = 'sum'
    
    # Provide default interval of 1800 seconds if not given as argument
    if interval is None:
        interval = 1800

    # Do not accept intervals larger than 84600 s (one day)
    if interval > 86400:
        print 'WARNING: Function event2time(): Interval larger than 86400 s not accepted.'
        print 'INTERVAL SET TO 86400 s (ONE DAY).\n'
        interval = 86400

    # Determine the start datetime of the new time series
    # Evaluate start time (first value in arrays)
    # First convert time of day to seconds
    startsecond = scipy.mod(doytime[0], 1) * 86400
    
    # Check what time the first interval in the regular time series would be
    starttime = scipy.floor(startsecond / interval) * interval
    # Increase to end of interval if it not exceeding one day (86400 s)
    if interval < 86400:
        starttime = starttime + interval 
    
    # Make sure to start on the day of installation
    if starttime > 86400:
        starttime = 86400
        print 'WARNING: interval exceeds past midnight of first day'
        print 'Start set to midnight of first day'
    start = starttime / 86400 + scipy.floor(doytime[0])
    
    # Determine end time
    endsecond = scipy.mod(doytime[len(doytime) - 1], 1) * 86400
    
    # Endtime is last full interval before the end of record
    endtime = scipy.floor(endsecond / interval) * interval
    end = endtime / 86400 + scipy.floor(doytime[len(doytime) - 1])
    
    # Determine start date and time, including year
    startdate = datetime.datetime(int(yyyy[0]), 1, 1, 0, 0, 0) + \
                datetime.timedelta(days= scipy.floor(start) - 1, \
                seconds=scipy.mod(start, 1) * 86400)
    
    # Determine the end date and time of time series
    enddate = datetime.datetime(int(yyyy[len(yyyy) - 1]), 1, 1, 0, 0, 0) + \
              datetime.timedelta(days=scipy.floor(end) - 1, \
              seconds=scipy.mod(end, 1) * 86400)
    
    # Create arrays for storing the equidistant time output
    YEAR = []
    DECTIME = []
    Y = []
    
    # Set initial date/time value to first date/time interval
    intervaldate = startdate
    
    # Set counters to zero
    i = 0 # desired time-index value counter
    j = 0 # event data index counter
    counter = 0 # counts events in one time interval
    
    # Set initial data value sum to zero
    processedY = 0
    
    # Start data processing  
    while intervaldate < enddate:
        i = i + 1
        # checkdate is event based date/time
        checkdate = datetime.datetime(int(yyyy[j]), 1, 1, 0, 0, 0) + \
                    datetime.timedelta(days=scipy.floor(doytime[j]) - 1, \
                    seconds=scipy.mod(doytime[j], 1) * 86400)
        # intervaldate is the date/time at regular interval
        intervaldate = intervaldate + datetime.timedelta(seconds=interval)
        # If the statement below is true, we have event(s)
        while checkdate < intervaldate:
            counter = counter + 1
            j = j + 1
            checkdate = datetime.datetime(int(yyyy[j]), 1, 1, 0, 0, 0) + \
                        datetime.timedelta(days=scipy.floor(doytime[j]) - 1, \
                        seconds=scipy.mod(doytime[j], 1) * 86400)
            processedY = processedY + X[j]
        # Append the time spaced data to the lists
        YEAR.append(intervaldate.year)
        # Recalculate to doy.decimaltime format
        dtime = (int(intervaldate.strftime("%j")) + \
                  (int(intervaldate.strftime("%H")) * 3600 + \
                   int(intervaldate.strftime("%M")) * 60 + \
                   int(intervaldate.strftime("%S"))) / 86400.0)
        # Correct of error in day when interval is 86400 s
        # This because new day starts at midnight, but processed data
        # covers previous day ending at midnight
        if interval == 86400:
            dtime = dtime - 1
        DECTIME.append(dtime)
        if method=='sum':
            Y.append(processedY)
        if method=='avg':
            # In case there are missing data in the interval, division by zero
            # could occur while averaging. In this case we indicate missing data
            # by -9999  
            if counter==0:
                counter=1.
                processedY=-9999
            Y.append(processedY/counter) 
        # Set processedY and counter to zero for next event
        processedY = 0
        counter = 0
        
    # Convert lists to arrays and output these arrays
    YEAR = scipy.array(YEAR)
    DECTIME = scipy.array(DECTIME)
    Y = scipy.array(Y)
    
    # Return year, doy.decimaltime and datavalue as output
    return YEAR, DECTIME, Y
    
def date2doy(dd=scipy.array([]),\
    mm=scipy.array([]),\
    yyyy=scipy.array([])):
    '''
    Function to calculate the julian day (day of year) from day,
    month and year.
        
    Input:
        - dd: (array of) day of month
        - mm: (array of) month
        - yyyy: (array of) year
        
    Output:
        - jd: (array of) julian day of year
        
    Examples:
        >>> date2doy(04,11,2006)
        308
        >>> date2doy(04,11,2008)
        309
        >>> day=[10,10,10]
        >>> month=[1,2,3]
        >>> year=[2007,2008,2009]
        >>> date2doy(day,month,year)
        array([ 10.,  41.,  69.])
        >>>
    '''
    # Determine length of array
    n = scipy.size(dd)
    if n < 2:   # Dealing with single value...date2doy
        # Determine julian day     
        doy = math.floor(275 * mm / 9 - 30 + dd) - 2
        if mm < 3:
            doy = doy + 2
        # Correct for leap years
        if (math.fmod(yyyy / 4.0, 1) == 0.0 and math.fmod(yyyy / 100.0, 1) != 0.0)\
        or math.fmod(yyyy / 400.0, 1) == 0.0:
            if mm > 2:
                doy = doy + 1
        doy = int(doy)
    else:   # Dealing with an array
        # Initiate the output array  
        doy = scipy.zeros(n)
        # Calculate julian days   
        for i in range(0, n):
            doy[i] = math.floor(275 * mm[i] / 9 - 30 + dd[i]) - 2
            if mm[i] < 3:
                doy[i] = doy[i] + 2
            # Correct for leap years
            if (math.fmod(yyyy[i] / 4.0, 1) == 0.0 and math.fmod(yyyy[i] / 100.0, 1)\
            != 0.0) or math.fmod(yyyy[i] / 400.0, 1) == 0.0:
                if mm[i] > 2:
                    doy[i] = doy[i] + 1
            doy[i] = int(doy[i])
    return doy # Julian day [integer]


def doy2date(doy=scipy.array([]),\
             yyyy=scipy.array([])):
    '''
    Function to calculate the date (dd, mm, yyyy) from Julian day and year.

    Input:
        - dd: (array of) day of month
        - mm: (array of) month
        - yy: (array of) year

    Output:
        - jd: (array of) julian day of year

    Examples:
        >>> doy2date(04,11,2006)
        308
        >>> doy2date(04,11,2008)
        309
        >>> day=[10,10,10]
        >>> month=[1,2,3]
        >>> year=[2007,2008,2009]
        >>> doy2date(day,month,year)
        array([ 10.,  41.,  69.])
    '''

    print 'This routine is under construction...'
    return

def sun_NR(doy=scipy.array([]),\
           lat=float):
    '''
    Function to calculate the maximum sunshine duration N and incoming radiation
    at the top of the atmosphere from day of year and latitude.

    NOTE: Only valid for latitudes between 0 and 67 degrees (tropics and
    temperate zone)

    Input:
        - doy: (array of) day of year
        - lat: latitude in degrees, negative for southern hemisphere

    Output:
        - N: (array of) maximum sunshine hours [h]
        - Rext: (array of) extraterrestrial radiation [J/day]  

    Examples:
        >>> sun_NR(50,60)
        308
        >>> jd_calc(04,11,2008)
        309
        >>> day=[10,10,10]
        >>> month=[1,2,3]
        >>> year=[2007,2008,2009]
        >>> jd_calc(day,month,year)
        array([ 10.,  41.,  69.])
    '''

    # Set solar constant [W/m2]
    S = 1367.0  #[W/m2]
    # Convert latitude [degrees] to radians
    latrad = lat * math.pi / 180.0
    # Determine length of doy array
    l = scipy.size(doy)
    # Check if we have a single value or an array
    if l < 2:   # Dealing with single value...
        # calculate solar declination dt [radians]
        dt = 0.409 * math.sin(2 * math.pi / 365 * doy - 1.39)
        # calculate sunset hour angle [radians]
        ws = scipy.arccos(-math.tan(latrad) * math.tan(dt))
        # Calculate sunshine duration N [h]
        N = 24 / math.pi * ws
        # Calculate day angle j [radians]
        j = 2 * math.pi / 365.25 * doy
        # Calculate relative distance to sun
        dr = 1.0 + 0.03344 * math.cos(j - 0.048869)
        # Calculate Rext
        Rext = S * 86400 / math.pi * dr * (ws * math.sin(latrad) * math.sin(dt)\
               + math.sin(ws) * math.cos(latrad) * math.cos(dt))
    else:   # Dealing with an array     
        # Initiate the output arrays
        N = scipy.zeros(l)
        Rext = scipy.zeros(l)
        dt = scipy.zeros(l)
        ws = scipy.zeros(l)
        j = scipy.zeros(l)
        dr = scipy.zeros(l)
        # Calculate N and Rext
        for i in range(0, l):       
            # calculate solar declination dt [radians]
            dt[i] = 0.409 * math.sin(2 * math.pi / 365 * doy[i] - 1.39)
            # calculate sunset hour angle [radians]
            ws[i] = scipy.arccos(-math.tan(latrad) * math.tan(dt[i]))
            # Calculate sunshine duration N [h]
            N[i] = 24 / math.pi * ws[i]
            # Calculate day angle j [radians]
            j[i] = 2 * math.pi / 365.25 * doy[i]
            # Calculate relative distance to sun
            dr[i] = 1.0 + 0.03344 * math.cos(j[i] - 0.048869)
            # Calculate Rext
            Rext[i] = S * 86400.0 / math.pi * dr[i] * (ws[i] * math.sin(latrad)\
                      * math.sin(dt[i]) + math.sin(ws[i]) * math.cos(latrad)\
                      * math.cos(dt[i]))
    # Convert to arrays
    N = scipy.array(N)
    Rext = scipy.array(Rext)
    return N, Rext

def es_calc(airtemp= scipy.array([])):
    '''
    Function to calculate saturated vapour pressure from temperature.
    For T<0 C:  Saturation vapour pressure equation for ice: Goff, J.A.,and S.
    Gratch, Low-pressure properties of water from \-160 to 212 F. 
    Transactions of the American society of
    heating and ventilating engineers, pp 95-122, presented
    at the 52nd annual meeting of the American society of
    heating and ventilating engineers, New York, 1946.
    
    For T>=0 C: Goff, J. A. Saturation pressure of water on the new Kelvin
    temperature scale, Transactions of the American
    society of heating and ventilating engineers, pp 347-354,
    presented at the semi-annual meeting of the American
    society of heating and ventilating engineers, Murray Bay,
    Quebec. Canada, 1957.
                
    Input:
        - airtemp: (array of) measured air temperature [Celsius]
        
    Output:
        - es: (array of) saturated vapour pressure [Pa]
        
    Examples:
        >>> es_calc(30.0)
        4242.7259946566316
        >>> x = [20, 25]
        >>> es_calc(x)
        array([ 2337.080198,  3166.824419])
    '''

    # Determine length of array
    n = scipy.size(airtemp)
    # Check if we have a single (array) value or an array
    if n < 2:
        # Calculate saturated vapour pressures, distinguish between water/ice
        if airtemp < 0:
            # Calculate saturation vapour pressure for ice
            log_pi = - 9.09718 * (273.16 / (airtemp + 273.15) - 1.0) \
                     - 3.56654 * math.log10(273.16 / (airtemp + 273.15)) \
                     + 0.876793 * (1.0 - (airtemp + 273.15) / 273.16) \
                     + math.log10(6.1071)
            es = math.pow(10, log_pi)   
        else:
            # Calculate saturation vapour pressure for water
            log_pw = 10.79574 * (1.0 - 273.16 / (airtemp + 273.15)) \
                     - 5.02800 * math.log10((airtemp + 273.15) / 273.16) \
                     + 1.50475E-4 * (1 - math.pow(10, (-8.2969 * ((airtemp +\
                     273.15) / 273.16 - 1.0)))) + 0.42873E-3 * \
                     (math.pow(10, (+4.76955 * (1.0 - 273.16\
                     / (airtemp + 273.15)))) - 1) + 0.78614
            es = math.pow(10, log_pw)
    else:   # Dealing with an array     
        # Initiate the output array
        es = scipy.zeros(n)
        # Calculate saturated vapour pressures, distinguish between water/ice
        for i in range(0, n):              
            if airtemp[i] < 0:
                # Saturation vapour pressure equation for ice
                log_pi = - 9.09718 * (273.16 / (airtemp[i] + 273.15) - 1.0) \
                         - 3.56654 * math.log10(273.16 / (airtemp[i] + 273.15)) \
                         + 0.876793 * (1.0 - (airtemp[i] + 273.15) / 273.16) \
                         + math.log10(6.1071)
                es[i] = math.pow(10, log_pi)
            else:
                # Calculate saturation vapour pressure for water  
                log_pw = 10.79574 * (1.0 - 273.16 / (airtemp[i] + 273.15)) \
                         - 5.02800 * math.log10((airtemp[i] + 273.15) / 273.16) \
                         + 1.50475E-4 * (1 - math.pow(10, (-8.2969\
                         * ((airtemp[i] + 273.15) / 273.16 - 1.0)))) + 0.42873E-3\
                         * (math.pow(10, (+4.76955 * (1.0 - 273.16\
                         / (airtemp[i] + 273.15)))) - 1) + 0.78614
                es[i] = pow(10, log_pw)
    # Convert from hPa to Pa
    es = es * 100.0
    return es # in Pa


def Delta_calc(airtemp= scipy.array([])):
    '''
    Function to calculate the slope of the temperature - vapour pressure curve
    (Delta) from air temperatures. Source: Technical regulations 49, World
    Meteorological Organisation, 1984. Appendix A. 1-Ap-A-3.
    
    Input:
        - airtemp: (array of) air temperature [Celsius]
    
    Output:
        - Delta: (array of) slope of saturated vapour curve [Pa K-1]
    
    Examples:
        >>> Delta_calc(30.0)
        243.34309166827097
        >>> x = [20, 25]
        >>> Delta_calc(x)
        array([ 144.665841,  188.625046])
    '''

    # Determine length of array
    n = scipy.size(airtemp)
    # Check if we have a single value or an array
    if n < 2:   # Dealing with single value...
        # calculate vapour pressure
        es = es_calc(airtemp) # in Pa
        # Convert es (Pa) to kPa
        es = es / 1000.0
        # Calculate Delta
        Delta = es * 4098.0 / math.pow((airtemp + 237.3), 2)*1000
    else:   # Dealing with an array         
        # Initiate the output arrays
        Delta = scipy.zeros(n)
        # calculate vapour pressure
        es = es_calc(airtemp) # in Pa
        # Convert es (Pa) to kPa
        es = es / 1000.0
        # Calculate Delta
        for i in range(0, n):
            Delta[i] = es[i] * 4098.0 / math.pow((airtemp[i] + 237.3), 2)*1000
    return Delta # in Pa/K


def ea_calc(airtemp= scipy.array([]),\
            rh= scipy.array([])):
    '''
    Function to calculate actual saturation vapour pressure.

    Input:
        - airtemp: array of measured air temperatures [Celsius]
        - rh: Relative humidity [%]

    Output:
        - ea: array of actual vapour pressure [Pa]

    Examples:
        >>> ea_calc(25,60)
        1900.0946514729308
    '''

    # Determine length of array
    n = scipy.size(airtemp)
    if n < 2:   # Dealing with single value...    
        # Calculate saturation vapour pressures
        es = es_calc(airtemp)
        # Calculate actual vapour pressure
        eact = float(rh) / 100.0 * es
    else:   # Dealing with an array
        # Initiate the output arrays
        eact = scipy.zeros(n)
        # Calculate saturation vapour pressures
        es = es_calc(airtemp)
        for i in range(0, n):
            # Calculate actual vapour pressure
            eact[i] = float(rh[i]) / 100.0 * es[i]
    return eact # in Pa


def vpd_calc(airtemp= scipy.array([]),\
             rh= scipy.array([])):
    '''
    Function to calculate vapour pressure deficit.

    Input:
        - airtemp: measured air temperatures [Celsius]
        - rh: (array of) rRelative humidity [%]
        
    Output:
        - vpd: (array of) vapour pressure deficits [Pa]
        
    Examples:
        >>> vpd_calc(30,60)
        1697.0903978626527
        >>> T=[20,25]
        >>> RH=[50,100]
        >>> vpd_calc(T,RH)
        array([ 1168.540099,   0.        ])
    '''

    # Determine length of array
    n = scipy.size(airtemp)
    # Check if we have a single value or an array
    if n < 2:   # Dealing with single value...
        # Calculate saturation vapour pressures
        es = es_calc(airtemp)
        eact = ea_calc(airtemp, rh) 
        # Calculate vapour pressure deficit
        vpd = es - eact
    else:   # Dealing with an array
        # Initiate the output arrays
        vpd = scipy.zeros(n)
        # Calculate saturation vapor pressures
        es = es_calc(airtemp)
        eact = ea_calc(airtemp, rh)
        # Calculate vapour pressure deficit
        for i in range(0, n):
            vpd[i] = es[i] - eact[i]
    return vpd # in hPa

def L_calc(airtemp= scipy.array([])):
    '''
    Function to calculate the latent heat of vapourisation,
    lambda, from air temperature. Source: J. Bringfelt. Test of a forest
    evapotranspiration model. Meteorology and Climatology Reports 52,
    SMHI, Norrkopping, Sweden, 1986.
    
    Input:
        - airtemp: (array of) air temperature [Celsius]
        
    Output:
        - L: (array of) lambda [J kg-1 K-1]
        
    Examples:
        >>> L_calc(25)
        2440883.8804624998
        >>> t
        [10, 20, 30]
        >>> L_calc(t)
        array([ 2476387.3842125,  2452718.3817125,  2429049.3792125])
    '''

    # Determine length of array
    n = scipy.size(airtemp)
    # Check if we have a single value or an array
    if n < 2:   # Dealing with single value...
        # Calculate lambda
        L = 4185.5 * (751.78 - 0.5655 * (airtemp + 273.15))
    else:   # Dealing with an array
       # Initiate the output arrays
        L = scipy.zeros(n)    
        # Calculate lambda
        for i in range(0, n):
            L[i] = 4185.5 * (751.78 - 0.5655 * (airtemp[i] + 273.15))
    return L # in J/kg


def cp_calc(airtemp= scipy.array([]),\
            rh= scipy.array([]),\
            airpress= scipy.array([])):
    '''
    Function to calculate the specific heat of air, c_p, from air temperatures, relative humidity and air pressure.
    
    Input:
        - airtemp: (array of) air temperature [Celsius]
        - rh: (array of) relative humidity data [%]
        - airpress: (array of) air pressure data [Pa]
        
    Output:
        cp: array of saturated c_p values [J kg-1 K-1]
        
    Examples:
        >>> cp_calc(25,60,101300)
        1014.0749457208065
        >>> t
        [10, 20, 30]
        >>> rh
        [10, 20, 30]
        >>> airpress
        [100000, 101000, 102000]
        >>> cp_calc(t,rh,airpress)
        array([ 1005.13411289,  1006.84399787,  1010.83623841])
    '''

    # Determine length of array
    n = scipy.size(airtemp)
    # Check if we have a single value or an array
    if n < 2:   # Dealing with single value...
        # calculate vapour pressures
        eact = ea_calc(airtemp, rh)
        # Calculate cp
        cp = 0.24 * 4185.5 * (1 + 0.8 * (0.622 * eact / (airpress - eact)))
    else:   # Dealing with an array
        # Initiate the output arrays
        cp = scipy.zeros(n)
        # calculate vapour pressures
        eact = ea_calc(airtemp, rh)
        # Calculate cp
        for i in range(0, n):
            cp[i] = 0.24 * 4185.5 * (1 + 0.8 * (0.622 * eact[i] / (airpress[i] - eact[i])))
    return cp # in J/kg/K


def gamma_calc(airtemp= scipy.array([]),\
               rh= scipy.array([]),\
               airpress=scipy.array([])):
    '''
    Function to calculate the psychrometric constant gamma.
    Source: J. Bringfelt. Test of a forest evapotranspiration model.
    Meteorology and Climatology Reports 52, SMHI, Norrköpping, Sweden,
    1986.
    
    Input:
        - airtemp: array of measured air temperature [Celsius]
        - rh: array of relative humidity values[%]
        - airpress: array of air pressure data [Pa]
        
    Output:
        - gamma: array of psychrometric constant values [Pa\K]
        
    Examples:
        >>> gamma_calc(10,50,101300)
        66.263433186572274
        >>> t
        [10, 20, 30]
        >>> rh
        [10, 20, 30]
        >>> airpress
        [100000, 101000, 102000]
        >>> gamma_calc(t,rh,airpress)
        array([ 65.255188,  66.656958,  68.242393])
    '''

    # Determine length of array
    n = scipy.size(airtemp)
    # Check if we have a single value or an array
    if n < 2:   # Dealing with single value...
        cp = cp_calc(airtemp, rh, airpress)
        L = L_calc(airtemp)
        # Calculate gamma
        gamma = cp * airpress / (0.622 * L)
    else:   # Dealing with an array
        # Initiate the output arrays
        gamma = scipy.zeros(n)
        # Calculate cp and Lambda values
        cp = cp_calc(airtemp, rh, airpress)
        L = L_calc(airtemp)
        # Calculate gamma
        for i in range(0, n):
            gamma[i] = cp[i] * airpress[i] / (0.622 * L[i])
    return gamma # in Pa\K


def rho_calc(airtemp= scipy.array([]),\
             rh= scipy.array([]),\
             airpress= scipy.array([])):
    '''
    Function to calculate the density of air, rho, from air
    temperatures, relative humidity and air pressure.
    
    Input:
        - airtemp: (array of) air temperature data [Celsius]
        - rh: (array of) relative humidity data [%]
        - airpress: (array of) air pressure data [Pa]
        
    Output:
        - rho: (array of) air density data [kg m-3]
        
    Examples:
        >>> t
        [10, 20, 30]
        >>> rh
        [10, 20, 30]
        >>> airpress
        [100000, 101000, 102000]
        >>> rho_calc(t,rh,airpress)
        array([ 1.22948419,  1.19787662,  1.16635358])
        >> rho_calc(10,50,101300)
        1.2431927125520903
    '''

    # Determine length of array
    n = scipy.size(airtemp)    
    # Check if we have a single value or an array
    if n < 2:   # Dealing with single value...
        eact = ea_calc(airtemp, rh)
        rho = 1.201 * (290.0 * (airpress - 0.378 * eact)) \
                 / (1000.0 * (airtemp + 273.15)) / 100.0
    else:   # Dealing with an array        
        # Initiate the output arrays
        rho = scipy.zeros(n)
        # Calculate actual vapour pressure
        eact = ea_calc(airtemp, rh)
        # calculate rho
        for i in range(0, n):
            rho[i] = 1.201 * (290.0 * (airpress[i] - 0.378 * eact[i])) \
                 / (1000.0 * (airtemp[i] + 273.15)) / 100.0
    return rho # in kg/m3


def pottemp(airtemp= scipy.array([]),\
            rh=scipy.array([]),\
            airpress=scipy.array([])):
    '''
    Function to calculate the potential temperature air, theta, from air
    temperatures, relative humidity and air pressure. Reference pressure
    1000 hPa.
    
    Input:
        - airtemp: (array of) air temperature data [Celsius]
        - rh: (array of) relative humidity data [%]
        - airpress: (array of) air pressure data [Pa]
        
    Output:
        - theta: (array of) potential air temperature data [Celsius]
        
    Examples:
        >>> t
        [5, 10, 20]
        >>> rh
        [45, 65, 89]
        >>> airpress
        [101300, 102000, 99800]
        >>> pottemp(t,rh,airpress)
        array([  3.97741582,   8.40874555,  20.16596828])
        >>> pottemp(5,45,101300)
        3.977415823848844
    '''

    # Determine length of array
    n = scipy.size(airtemp)
    # Check if we have a single value or an array
    if n < 2:   # Dealing with single value...
        # Determine cp
        cp = cp_calc(airtemp, rh, airpress)
        theta = (airtemp + 273.15) * pow((100000.0 / airpress), \
                                       (287.0 / cp)) - 273.15
    else:   # Dealing with an array
        # Initiate the output array  
        theta = scipy.zeros(n)
        # Determine cp
        cp = cp_calc(airtemp, rh, airpress)
        # Calculate potential temperature   
        for i in range(0, n):
            theta[i] = ((airtemp[i] + 273.15) * math.pow((100000.0 / airpress[i]), \
                       (287.0 / cp[i])) - 273.15)
    return theta # in degrees celsius

def windvec(u= scipy.array([]),\
            D=scipy.array([])):
    '''
    Function to calculate the wind vector from time series of wind
    speed and direction 
    
    Input:
        - u: array of wind speeds [m/s]
        - D: array of wind directions [degrees from North]
        
    Output:
        - uv: Vector wind speed [m/s]
        - Dv: Vector wind direction [degrees from North]
        
    Examples:
        >>> u
        array([[ 3. ],
               [ 7.5],
               [ 2.1]])
        >>> D
        array([[340],
               [356],
               [  2]])
        >>> windvec(u,D)
        (4.1623542028369052, 353.21188820369366)
        >>> uv,Dv=windvec(u,D)
        >>> uv
        4.1623542028369052
        >>> Dv
        353.21188820369366
    '''

    ve = 0.0 # define east component of wind speed
    vn = 0.0 # define north component of wind speed
    D = D * math.pi / 180.0 # convert wind direction degrees to radians
    for i in range(0, len(u)):
        ve = ve + u[i] * math.sin(D[i]) # calculate sum east speed components
        vn = vn + u[i] * math.cos(D[i]) # calculate sum north speed components
    ve = - ve / len(u) # determine average east speed component
    vn = - vn / len(u) # determine average north speed component
    uv = math.sqrt(ve * ve + vn * vn) # calculate wind speed vector magnitude
    # Calculate wind speed vector direction
    vdir = scipy.arctan2(ve, vn)
    vdir = vdir * 180.0 / math.pi # Convert radians to degrees
    if vdir < 180:
        Dv = vdir + 180.0
    else:
        if vdir > 180.0:
            Dv = vdir - 180
        else:
            Dv = vdir
    return uv, Dv # uv in m/s, Dv in dgerees from North
