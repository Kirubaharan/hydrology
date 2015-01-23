# -*- coding: utf-8 -*-
""" Evaplib: A libray with Python functions for calculation of 
    evaporation from meteorological data.

    Functions:
    
        - penman: Calculate Penman (1948, 1956) open water evaporation
        - makkink: Calculate evaporation according to Makkink (1965)
        - Ept: Calculate evaporation according to Priestley and Taylor (1972)
        - ET0pm: Calculate Penman Monteith reference evaporation short grass
        - Epm: Calculate Penman-Monteith evaporation (actual)
        - ra: Calculate aerodynamic resistance from windspeed and
          roughnes parameters
        - tvardry: calculate sensible heat flux from temperature variations
          (Vugts et al., 1993)
        - gash79: Gash (1979) analytical rainfall interception model
"""
__author__ = "Maarten J. Waterloo <maarten.waterloo@falw.vu.nl>"
__version__ = "1.0"
__date__ = "Sep 2012"

# Make a help entry for this library
def evaplib():
    """ A libray with Python functions for calculation of 
    evaporation from meteorological data.

    Functions:
    
        - E0: Calculate Penman (1948, 1956) open water evaporation 
        - Em: Calculate evaporation according to Makkink (1965)
        - Ept: Calculate evaporation according to Priestley and Taylor (1972)
        - ET0pm: Calculate Penman Monteith reference evaporation short grass (FAO)
        - Epm: Calculate Penman Monteith reference evaporation (Monteith, 1965)
        - ra: Calculate  from windspeed and roughnes parameters
        - tvardry: calculate sensible heat flux from temperature variations
          (Vugts et al., 1993)
        - gash79: calculate rainfall interception (Gash, 1979)
          
    Author: Maarten J. Waterloo <m.j.waterloo@vu.nl>
    Version 1.0
    Date: Sep 2012
    """
    print 'A libray with Python functions for calculation of'
    print 'evaporation from meteorological and vegetation data.\n\n'
    print 'Functions:\n'
    print '- E0: Calculate Penman (1948, 1956) open water evaporation'
    print '- Em: Calculate evaporation according to Makkink (1965)'
    print '- Ept: Calculate evaporation according to Priestley and Taylor (1972).'
    print '- ET0pm: Calculate Penman Monteith reference evaporation short grass.'
    print '- Epm: Calculate Penman Monteith evaporation (Monteith, 1965).'
    print '- ra: Calculate aerodynamic resistance.'
    print '- tvardry: calculate sensible heat flux from temperature variations \
          (Vugts et al., 1993).'
    print '- gash79: calculate rainfall interception (Gash, 1979).\n\n'
    print 'Author: ',__author__
    print 'Version: ',__version__ 
    print 'Date: ',__date__
    return

# First load python micrometeorological functions
import meteolib
import scipy

'''
    ================================================================
    Potential evaporation functions
    ================================================================  
'''

def E0(airtemp = scipy.array([]),\
       rh = scipy.array([]),\
       airpress = scipy.array([]),\
       Rs = scipy.array([]),\
       # N = scipy.array([]),\
       Rext = scipy.array([]),\
       u = scipy.array([]),\
       Z=0.0):
    '''
    Function to calculate daily Penman open water evaporation (in mm/day).
    Equation according to J.D. Valiantzas (2006). Simplified versions
    for the Penman evaporation equation using routine weather data.
    J. Hydrology 331: 690-702. Following Penman (1948,1956). Albedo set
    at 0.06 for open water.
    
    Input (measured at 2 m height):
        - airtemp: (array of) daily average air temperatures [Celsius]
        - rh: (array of) daily average relative humidity [%]
        - airpress: (array of) daily average air pressure data [Pa]
        - Rs: (array of) daily incoming solar radiation [J/m2/day]
        - N: (array of) maximum daily sunshine hours [h]
        - Rext: (array of) daily extraterrestrial radiation [J/m2/day]
        - u: (array of) daily average wind speed at 2 m [m/s]
        - Z: (array of) site elevation [m a.s.l.], default is zero...
   
    Output:
        - E0: (array of) Penman open water evaporation values [mm/day]

    Examples:
        >>> # T, RH, etc. are arrays of data...
        >>> E0_data = E0(T,RH,press,Rs,N,Rext,u) # for zero elevation
        >>> E0_data = E0(T,RH,press,Rs,N,Rext,u,1000.0) # at 1000 m a.s.l
        >>> # With single values and default elevation...
        >>> E0(20.67,67.0,101300.0,22600000,14.4,42000000,2.0)
        6.3102099052283389
        >>> # for elevation of 1.0 m a.s.l.
        >>> E0(21.65,67.0,101300.0,24200000,14.66,42200000,1.51,1.0)
        6.5991748573832343
        >>> 
    '''
    # Set constants
    albedo = 0.06 # Open water albedo
    sigma = 4.903E-3 # Stefan Boltzmann constant J/m2/K4/d
    # Calculate Delta, gamma and lambda
    DELTA = meteolib.Delta_calc(airtemp) # [Pa/K]
    gamma = meteolib.gamma_calc(airtemp,rh,airpress) # [Pa/K]
    Lambda = meteolib.L_calc(airtemp) # [J/kg]
    # Calculate saturated and actual water vapour pressures
    es = meteolib.es_calc(airtemp) # [Pa]
    ea = meteolib.ea_calc(airtemp,rh) # [Pa]
    # Determine length of array
    l = scipy.size(airtemp)
    # Check if we have a single value or an array
    if l < 2:   # Dealing with single value...
        Rns = (1.0-albedo)*Rs # Shortwave component [J/m2/d]
        # Calculate clear sky radiation Rs0 
        Rs0 = (0.75+2E-5*Z)*Rext
        f = 1.35*Rs/Rs0-0.35
        epsilom = 0.34-0.14*scipy.sqrt(ea/1000)
        Rnl = f*epsilom*sigma*(airtemp+273.15)**4 # Longwave component [J/m2/d]
        Rnet = Rns-Rnl # Net radiation [J/m2/d]
        Ea = (1+0.536*u)*(es/1000.-ea/1000.)
        E0 = DELTA/(DELTA+gamma)*Rnet/Lambda+gamma/(DELTA+gamma)*6430000*Ea/Lambda
    else:   # Dealing with an array  
        # Initiate output arrays
        E0 = scipy.zeros(l)
        Rns = scipy.zeros(l)
        Rs0 = scipy.zeros(l)
        f = scipy.zeros(l)
        epsilom = scipy.zeros(l)
        Rnl = scipy.zeros(l)
        Rnet = scipy.zeros(l)
        Ea = scipy.zeros(l)
        for i in range(0,l):
            # calculate longwave radiation component Rln (J/m2/day)
            Rns[i] = (1.0-albedo)*Rs[i] # Shortwave component [J/m2/d]
            # Calculate clear sky radiation Rs0 
            Rs0[i] = (0.75+2E-5*Z)*Rext[i]
            f[i] = 1.35*Rs[i]/Rs0[i]-0.35
            epsilom[i] = 0.34-0.14*scipy.sqrt(ea[i]/1000)
            Rnl[i] = f[i]*epsilom[i]*sigma*(airtemp[i]+273.15)**4 # Longwave component [J/m2/d]
            Rnet[i] = Rns[i]-Rnl[i] # Net radiation [J/m2/d]
            Ea[i] = (1+0.536*u[i])*(es[i]/1000-ea[i]/1000)
            E0[i] = DELTA[i]/(DELTA[i]+gamma[i])*Rnet[i]/Lambda[i]+gamma[i]/(DELTA[i]+gamma[i])* \
                6430000*Ea[i]/Lambda[i]
    return E0

def ET0pm(airtemp = scipy.array([]),\
          rh = scipy.array([]),\
          airpress = scipy.array([]), \
          Rs = scipy.array([]),\
          N = scipy.array([]),\
          Rext = scipy.array([]),\
          u = scipy.array([]), \
          Z=0.0):
    '''
    Function to calculate daily Penman Monteith reference evaporation
    (in mm/day). Source: R.G. Allen, L.S. Pereira, D. Raes and M. Smith
    (1998). Crop evapotranspiration - Guidelines for computing crop
    water requirements - FAO Irrigation and drainage paper 56. FAO -
    Food and Agriculture Organization of the United Nations, Rome, 1998 
    
    Input (measured at 2 m height):
        - airtemp: (array of) daily average air temperatures [Celsius]
        - rh: (array of) daily average relative humidity values[%]
        - airpress: (array of) daily average air pressure data [hPa]
        - Rs: (array of) total incoming shortwave radiation [J/m2/day]
        - N: daylength [h]
        - Rext: Incoming shortwave radiation at the top of the atmosphere [J/m2/day]
        - u: windspeed [m/s]
        - Z: elevation [m], default is 0.0 m
   
    Output:
        - ET0pm: (array of) Penman Monteith reference evaporation (short grass with optimum water supply) values [mm] 

    Examples:--
        >>> Eref_data = ET0pm(T,RH,press,Rs,N,Rext,u)    
    '''
    # Set constants
    albedo = 0.23 # short grass albedo
    sigma = 4.903E-3 # Stefan Boltzmann constant J/m2/K4/d
    # Calculate Delta, gamma and lambda
    DELTA = meteolib.Delta_calc(airtemp) # [Pa/K]
    gamma = meteolib.gamma_calc(airtemp,rh,airpress) # [Pa/K]
    Lambda = meteolib.L_calc(airtemp) # [J/kg]
    # Calculate saturated and actual water vapour pressures
    es = meteolib.es_calc(airtemp) # [Pa]
    ea = meteolib.ea_calc(airtemp,rh) # [Pa]
    # Determine length of array
    l = scipy.size(airtemp)
    # Check if we have a single value or an array
    if l < 2:   # Dealing with single value...
        Rns = (1.0-albedo)*Rs # Shortwave component [J/m2/d]
        # Calculate clear sky radiation Rs0 
        Rs0 = (0.75+2E-5*Z)*Rext # Clear sky radiation [J/m2/d]
        f = 1.35*Rs/Rs0-0.35
        epsilom = 0.34-0.14*scipy.sqrt(ea/1000)
        Rnl = f*epsilom*sigma*(airtemp+273.15)**4 # Longwave component [J/m2/d]
        Rnet = Rns-Rnl # Net radiation [J/m2/d]
        ET0pm = (DELTA/1000.*Rnet/Lambda+900./(airtemp+273.16)*u*(es-ea)/1000\
                 *gamma/1000)/(DELTA/1000.+gamma/1000*(1.+0.34*u))
    else:   # Dealing with an array  
        # Initiate output arrays
        ET0pm = scipy.zeros(l)
        Rns = scipy.zeros(l)
        Rs0 = scipy.zeros(l)
        f = scipy.zeros(l)
        epsilom = scipy.zeros(l)
        Rnl = scipy.zeros(l)
        Rnet = scipy.zeros(l)
        for i in range(0,l):
            # calculate longwave radiation component Rln (J/m2/day)
            Rns[i] = (1.0-albedo)*Rs[i] # Shortwave component [J/m2/d]
            # Calculate clear sky radiation Rs0 
            Rs0[i] = (0.75+2E-5*Z)*Rext[i]
            f[i] = 1.35*Rs[i]/Rs0[i]-0.35
            epsilom[i] = 0.34-0.14*scipy.sqrt(ea[i]/1000)
            Rnl[i] = f[i]*epsilom[i]*sigma*(airtemp[i]+273.15)**4 # Longwave component [J/m2/d]
            Rnet[i] = Rns[i]-Rnl[i] # Net radiation [J/m2/d]
            ET0pm[i] = (DELTA[i]/1000.*Rnet[i]/Lambda[i]+900./(airtemp[i]+273.16)* \
                       u[i]*(es[i]-ea[i])/1000*gamma[i]/1000)/ \
                      (DELTA[i]/1000.+gamma[i]/1000*(1.+0.34*u[i]))
    return ET0pm # FAO reference evaporation [mm/day]


def Em(airtemp = scipy.array([]),\
       rh = scipy.array([]),\
       airpress = scipy.array([]),\
       Rs = scipy.array([])):
    '''
    Function to calculate Makkink evaporation (in mm/day). The Makkink
    evaporation is a reference crop evaporation used in the Netherlands,
    which is combined with a crop factor to provide an estimate of actual
    crop evaporation. Source: De Bruin, H.A.R.,1987. From Penman to
    Makkink', in Hooghart, C. (Ed.), Evaporation and Weather, Proceedings
    and Information. Comm. Hydrological Research TNO, The Hague. pp. 5-30.

    
    Input (measured at 2 m height):
        - airtemp: (array of) daily average air temperatures [Celsius]
        - rh: (array of) daily average relative humidity values[%]
        - airpress: (array of) daily average air pressure data [Pa]
        - Rs: (array of) average daily incoming solar radiation [J/m2/day]
   
    Output:
        - Em: (array of) Makkink evaporation values [mm]

    Examples:
        >>> Em_data = Em(T,RH,press,Rs)
        >>> Em(21.65,67.0,101300,24200000)
        4.5038304791979913
    '''
    # Calculate Delta and gamma constants
    DELTA = meteolib.Delta_calc(airtemp)
    gamma = meteolib.gamma_calc(airtemp,rh,airpress)
    Lambda = meteolib.L_calc(airtemp)
    # Determine length of array
    l = scipy.size(airtemp)
    # Check if we have a single value or an array
    if l < 2:   # Dealing with single value...
        # calculate Em [mm/day]
        Em = 0.65 * DELTA/(DELTA + gamma) * Rs / Lambda
    else:   # Dealing with an array         
        # Initiate output array
        Em = scipy.zeros(l)
        for i in range(0,l):   
            # calculate Em [mm/day]
            Em[i]= 0.65*DELTA[i]/(DELTA[i]+gamma[i])*Rs[i]/Lambda[i]
        Em=scipy.array(Em)
    return Em


def Ept(airtemp = scipy.array([]),\
        rh = scipy.array([]),\
        airpress = scipy.array([]),\
        Rn = scipy.array([]),\
        G = scipy.array([])):
    '''
    Function to calculate daily Priestley - Taylor evaporation (in mm).
    Source: Priestley, C.H.B. and R.J. Taylor, 1972. On the assessment
    of surface heat flux and evaporation using large-scale parameters.
    Mon. Weather Rev. 100:81-82.
    
    Input (measured at 2 m height):
        - airtemp: (array of) daily average air temperatures [Celsius]
        - rh: (array of) daily average relative humidity values[%]
        - airpress: (array of) daily average air pressure data [Pa]
        - Rn: (array of) average daily net radiation [J/m2/day]
        - G: (array of) average daily soil heat flux [J/m2/day]
   
    Output:
        - Ept: (array of) Priestley Taylor evaporation values [mm]

    Examples:
        >>> Ept_data = Ept(T,RH,press,Rn,G)
        >>> Ept(21.65,67.0,101300,18200000,600000)
        6.3494561161280778
    '''
    # Calculate Delta and gamma constants
    DELTA = meteolib.Delta_calc(airtemp)
    gamma = meteolib.gamma_calc(airtemp,rh,airpress)
    Lambda = meteolib.L_calc(airtemp)
    # Determine length of array
    l = scipy.size(airtemp)
    # Check if we have a single value or an array
    if l < 2:   # Dealing with single value...
        # calculate Em [mm/day]
        Ept= 1.26*DELTA/(DELTA+gamma)*(Rn-G)/Lambda
    else:   # Dealing with an array         
        # Initiate output array
        Ept = scipy.zeros(l)
        for i in range(0,l):   
            # calculate Ept [mm/day]
            Ept[i]= 1.26*DELTA[i]/(DELTA[i]+gamma[i])*(Rn[i]-G[i])/Lambda[i]
        Ept= scipy.array(Ept)
    return Ept

'''
    ================================================================
    Actual evaporation functions and utilities
    ================================================================  
'''

def ra(z=float,\
       z0=float,\
       d=float,\
       u = scipy.array([])):
    '''
    Function to calculate the aerodynamic resistance 
    (in s/m) from windspeed and height/roughness values
    
    Input (measured at 2 m height):
        - z: measurement height [m]
        - z0: roughness length [m]
        - d: displacement length [m]
        - u: (array of) windspeed [m/s]


    Output:
        - ra: (array of) aerodynamic resistances [s/m]

    Examples:
        >>> ra(3,0.12,2.4,5.0)
        3.2378629924752942
        >>> u=([2,4,6])
        >>> ra(3,0.12,2.4,u)
        array([ 8.09465748,  4.04732874,  2.69821916])
    '''
    # Determine length of array
    l = scipy.size(u)
    # Check if we have a single value or an array
    if l < 2:   # Dealing with single value...
        ra= (scipy.log((z-d)/z0))**2/(0.16*u)
    else:   # Dealing with an array  
        # Initiate output arrays
        ra = scipy.zeros(l)
        for i in range(0,l):
            ra[i]= (scipy.log((z-d)/z0))**2/(0.16*u[i])
    return ra # aerodynamic resistanc in s/m

def Epm(airtemp = scipy.array([]),\
        rh = scipy.array([]),\
        airpress = scipy.array([]),\
        Rn = scipy.array([]),\
        G = scipy.array([]),\
        ra = scipy.array([]),\
        rs = scipy.array([])):
    '''
    Function to calculate the Penman Monteith evaporation
    (in mm) Monteith, J.L. (1965) Evaporation and environment.
    Symp. Soc. Exp. Biol. 19, 205-224
    
    Input (measured at 2 m height):
        - airtemp: (array of) daily average air temperatures [C]
        - rh: (array of) daily average relative humidity values[%]
        - airpress: (array of) daily average air pressure data [hPa]
        - Rn: (array of) average daily net radiation [J]
        - G: (array of) average daily soil heat flux [J]
        - ra: aerodynamic resistance [s/m]
        - rs: surface resistance [s/m]

    Output:
        - Epm: (array of) Penman Monteith evaporation values [mm]
    
    Examples:
        >>> Epm_data = Epm(T,RH,press,Rn,G,ra,rs)    
    '''
    # Calculate Delta, gamma and lambda
    DELTA = meteolib.Delta_calc(airtemp)/100. # [hPa/K]
    airpress=airpress*100. # [Pa]
    gamma = meteolib.gamma_calc(airtemp,rh,airpress)/100. # [hPa/K]
    Lambda = meteolib.L_calc(airtemp) # [J/kg]
    rho = meteolib.rho_calc(airtemp,rh,airpress)
    cp = meteolib.cp_calc(airtemp,rh,airpress)
    # Calculate saturated and actual water vapour pressures
    es = meteolib.es_calc(airtemp)/100. # [hPa]
    ea = meteolib.ea_calc(airtemp,rh)/100. # [hPa]
    # Determine length of array
    l = scipy.size(airtemp)
    # Check if we have a single value or an array
    if l < 2:   # Dealing with single value...
        Epm = (DELTA*Rn+rho*cp*(es-ea)*ra/(DELTA+gamma*(1.+rs/ra)))/Lambda
    else:   # Dealing with an array  
        # Initiate output arrays
        Epm = scipy.zeros(l)
        for i in range(0,l):
            Epm = (DELTA[i]*Rn[i]+rho[i]*cp[i]*(es[i]-ea[i])*ra[i]/(DELTA[i] \
                + gamma[i]*(1.+rs[i]/ra[i])))/Lambda[i]
    return Epm # actual ET in mm
    
    
def tvardry(rho = scipy.array([]),\
    cp = scipy.array([]),\
    T = scipy.array([]),\
    sigma_t = scipy.array([]),\
    z= float(),\
    d= 0.0):
    '''Function to calculate the sensible heat flux (H, in W/m2) from high
    frequency temperature measurements and its standard deviation. 
    Source: H.F. Vugts, M.J. Waterloo, F.J. Beekman, K.F.A. Frumau and L.A.
    Bruijnzeel. The temperature variance method: a powerful tool in the
    estimation of actual evaporation rates. In J. S. Gladwell, editor, 
    Hydrology of Warm Humid Regions, Proc. of the Yokohama Symp., IAHS
    Publication No. 216, pages 251-260, July 1993.
    
    NOTE: This function holds only for free convective conditions when C2*z/L
    >>1, where L is the Obhukov length.
    
    Input:
        - rho: (array of) air density values [kg m-3]
        - cp: (array of) specific heat at constant temperature values [J kg-1 K-1]
        - T: (array of) temperature data [Celsius]
        - sigma_t: (array of) standard deviation of temperature data [Celsius]
        - z: temperature measurement height above the surface [m]
        - d: displacement height due to vegetation, default is zero [m]
        
    Output:
        - H: (array of) sensible heat flux [W/m2]
        
    Example:
        >>> H=tvardry(rho,cp,T,sigma_t,z,d)
        >>> H
        35.139511191461651
        >>>
    '''
    k = 0.40 # von Karman constant
    g = 9.81 # acceleration due to gravity [m/s^2]
    C1 =  2.9 # De Bruin et al., 1992
    C2 = 28.4 # De Bruin et al., 1992
    # L= Obhukov-length [m]
     
    #Free Convection Limit
    H = rho * cp * scipy.sqrt((sigma_t/C1)**3 * k * g * (z-d) / (T+273.15) * C2)
    #else:
    # including stability correction
    #zoverL = z/L
    #tvardry = rho * cp * scipy.sqrt((sigma_t/C1)**3 * k*g*(z-d) / (T+273.15) *\
    #          (1-C2*z/L)/(-1*z/L))
    
    #Check if we get complex numbers (square root of negative value) and remove 
    #I = find(zoL >= 0 | H.imag != 0);
    #H(I) = scipy.ones(size(I))*NaN;
        
    return H # sensible heat flux

'''
    ================================================================
    Rainfall interception functions and utilities
    ================================================================  
'''

def gash79(Pg=scipy.array([]),
            ER=float,
            S=float,
            St=float,
            p=float,       
            pt=float):
    
    '''
    This is the help text: Function to calculate precipitation interception loss
    according to J.H.C. Gash, An analytical model of rainfall interception by
    forests, Quarterly Journal of the Royal Meteorological Society, 1979, 105,
    pp. 43--55,
    
    Input:
        - Pg: daily rainfall data [mm]
        - ER: evaporation percentage of total rainfall [mm/h]
        - S: storage capacity canopy [mm]
        - St: stem storage capacity [mm]
        - p: direct throughfall [mm]
        - pt: stem precipitation [mm]
    
    Output:
        - date: 
        - Pg: Daily rainfall [mm]
        - Ei: Interception [mm]
        - TF: through fall [mm]
        - SF: stemflow [mm]
        
    Examples:
        >>> gash79(Pg, ER, S, St, p, pt)
    '''
    
    # Determine length of array Pg
    l = scipy.size(Pg)
    # Check if we have a single precipitation value or an array
    if l < 2:   # Dealing with single value...
        
        #PGsat calculation (for the saturation of the canopy)
        PGsat = -(1/ER*S)* scipy.log((1-(ER/(1-p-pt))))

        #Set initial values to zero
        Ecan= 0.
        Etrunk= 0.

        # Calculate interception for different storm sizes
        if (Pg<PGsat and Pg>0): 
            Ecan=(1-p-pt)*Pg
            if (Pg>St/pt):
                Etrunk=St+pt*Pg
            Ei = Ecan+Etrunk
        if (Pg>PGsat and Pg<St/pt):
            Ecan=((((1-p-Pt)*PGsat)-S) + (ER*(Pg-PGsat)) + S)
            Etrunk=0.
            Ei = Ecan+Etrunk
        if (Pg>PGsat and Pg>(St/pt)):
            Ecan=((((1-p-pt)*PGsat)-S)+ (ER*(Pg-PGsat)) + S+(St+pt*Pg))
            Etrunk=St+pt*Pg
        Ei = Ecan + Etrunk
        TF = Pg-Ei
        SF=0
        
    else:
        #Define variables and constants
        n = scipy.size(Pg)
        TF = scipy.zeros(n)
        SF = scipy.zeros(n)
        Ei = scipy.zeros(n)
        Etrunk = scipy.zeros(n)

        #Set results to zero if rainfall Pg is zero
        TF[Pg==0]=0.
        SF[Pg==0]=0.
        Ei[Pg==0]=0.
        Etrunk[Pg==0]=0.

        #PGsat calc (for the saturation of the canopy)
        PGsat = -(1/ER*S)* scipy.log((1-(ER/(1-p-pt))))

        #Process rainfall series
        for i in range (0,n):
            Ecan= 0.
            Etrunk= 0.
            if (Pg[i]<PGsat and Pg[i]>0): 
                Ecan=(1-p-pt)*Pg[i]
                if (Pg[i]>St/pt):
                    Etrunk=St+pt*Pg[i]
                Ei[i]=Ecan+Etrunk
            if (Pg[i]>PGsat and Pg[i]<St/pt):
                Ecan=((((1-p-Pt)*PGsat)-S)+ (ER*(Pg[i]-PGsat)) + S)
                Etrunk=0.
                Ei[i]
            if (Pg[i]>PGsat and Pg[i]>(St/Pt)):
                Ecan=((((1-p-Pt)*PGsat)-S)+ (ER*(Pg[i]-PGsat)) + S+(St+Pt*Pg[i]))
                Etrunk=St+pt*Pg[i]
            Ei[i]=Ecan+Etrunk
            TF[i]=Pg[i]-Ei[i]
    return Pg, TF, SF, Ei 