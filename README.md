#Hydrology

##My Hydrology  repository
I use Python and R for managing my workflow in hydrological modelling. This repository is a collection of those codes. You are welcome to use and modify it for your purpose. See [Licence](../master/LICENSE) for terms and conditions.
My contact details are at the bottom of this page.
###Tutorial for stage volume
See the [Stage - Volume Tutorial](../master/stage_volume_tutorial.py).
Check out the comments.  

###3D plotting
For 3D plotting of interpolated stream profile  [Profile Creator](../master/profile_creator.py).

###591 Check dam profile
#### Stage - Surface Area relationship
The stage(water height) vs water surface area relationship is calculated in this file [591 Check Dam](../master/profile_creator_591.py).
This file does following functions:
 1. Fills in between profiles.
 2. Creates x,y,z grid from profile.
 3. Creates interpolation of uniform grid.
 4. Creates a contour and 3D surface plot from the interpolated data.
 5. Calculates the contour area for given elevation levels.
 6. Plot of surface area vs stage.

### Markov chain Monte Carlo(MCMC) estimation of missing wind speed
Use of MCMC to fill missing wind speed values in weather data. [MCMC](../master/wind_speed_had_mcmc.py) 

###Contact
For related queries please mail: haran.kiruba@gmail.com

![Contact](http://i.imgur.com/C9rENMG.png)