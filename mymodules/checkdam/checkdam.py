__author__ = 'kiruba'
#function to create stage volume output

def calcvolume(profile, order, dy):
    """
    Profile = df.Y1,df.Y2,.. and order = 1,2,3
    :param profile: series of Y values
    :param order: distance from origin
    :param dy: thickness of profile in m
    :return: volume for profile
    """

    results = []

    for z in dz:
        water_area = 0
        for y1, y2 in pairwise(profile):
            delev = (y2 - y1) / 10
            elev = y1
            for b in range(1, 11, 1):
                elev += delev
                if z > elev:
                    water_area += (0.1 * (z-elev))
        calc_vol = water_area * dy
        results.append(calc_vol)
    output[('Volume_%s' % order)] = results
