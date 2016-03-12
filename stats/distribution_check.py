#!/usr/bin/env python
#title           :distribution_checkX.py
#description     :Checks a sample against 80 distributions by applying the Kolmogorov-Smirnov test.
#author          :Andre Dietrich
#email           :dietrich@ivs.cs.uni-magdeburg.de
#date            :07.10.2014
#version         :0.1
#usage           :python distribution_check.py -f filename -v
#python_version  :2.* and 3.*
#########################################################################################
from __future__ import print_function

import scipy.stats
import operator
import warnings
import random
import math
# just for surpressing warnings
warnings.simplefilter('ignore')

from joblib import Parallel, delayed
import multiprocessing
from optparse import OptionParser

########################################################################################

# list of all available distributions
cdfs = {
    "alpha": {"p":[], "D": []},           #Alpha
    "anglit": {"p":[], "D": []},          #Anglit
    "arcsine": {"p":[], "D": []},         #Arcsine
    "beta": {"p":[], "D": []},            #Beta
    "betaprime": {"p":[], "D": []},       #Beta Prime
    "bradford": {"p":[], "D": []},        #Bradford
    "burr": {"p":[], "D": []},            #Burr
    "cauchy": {"p":[], "D": []},          #Cauchy
    "chi": {"p":[], "D": []},             #Chi
    "chi2": {"p":[], "D": []},            #Chi-squared
    "cosine": {"p":[], "D": []},          #Cosine
    "dgamma": {"p":[], "D": []},          #Double Gamma
    "dweibull": {"p":[], "D": []},        #Double Weibull
    "erlang": {"p":[], "D": []},          #Erlang
    "expon": {"p":[], "D": []},           #Exponential
    "exponweib": {"p":[], "D": []},       #Exponentiated Weibull
    "exponpow": {"p":[], "D": []},        #Exponential Power
    "f": {"p":[], "D": []},               #F (Snecdor F)
    "fatiguelife": {"p":[], "D": []},     #Fatigue Life (Birnbaum-Sanders)
    "fisk": {"p":[], "D": []},            #Fisk
    "foldcauchy": {"p":[], "D": []},      #Folded Cauchy
    "foldnorm": {"p":[], "D": []},        #Folded Normal
    "frechet_r": {"p":[], "D": []},       #Frechet Right Sided, Extreme Value Type II
    "frechet_l": {"p":[], "D": []},       #Frechet Left Sided, Weibull_max
    "gamma": {"p":[], "D": []},           #Gamma
    "gausshyper": {"p":[], "D": []},      #Gauss Hypergeometric
    "genexpon": {"p":[], "D": []},        #Generalized Exponential
    "genextreme": {"p":[], "D": []},      #Generalized Extreme Value
    "gengamma": {"p":[], "D": []},        #Generalized gamma
    "genhalflogistic": {"p":[], "D": []}, #Generalized Half Logistic
    "genlogistic": {"p":[], "D": []},     #Generalized Logistic
    "genpareto": {"p":[], "D": []},       #Generalized Pareto
    "gilbrat": {"p":[], "D": []},         #Gilbrat
    "gompertz": {"p":[], "D": []},        #Gompertz (Truncated Gumbel)
    "gumbel_l": {"p":[], "D": []},        #Left Sided Gumbel, etc.
    "gumbel_r": {"p":[], "D": []},        #Right Sided Gumbel
    "halfcauchy": {"p":[], "D": []},      #Half Cauchy
    "halflogistic": {"p":[], "D": []},    #Half Logistic
    "halfnorm": {"p":[], "D": []},        #Half Normal
    "hypsecant": {"p":[], "D": []},       #Hyperbolic Secant
    "invgamma": {"p":[], "D": []},        #Inverse Gamma
    "invgauss": {"p":[], "D": []},        #Inverse Normal
    "invweibull": {"p":[], "D": []},      #Inverse Weibull
    "johnsonsb": {"p":[], "D": []},       #Johnson SB
    "johnsonsu": {"p":[], "D": []},       #Johnson SU
    "laplace": {"p":[], "D": []},         #Laplace
    "logistic": {"p":[], "D": []},        #Logistic
    "loggamma": {"p":[], "D": []},        #Log-Gamma
    "loglaplace": {"p":[], "D": []},      #Log-Laplace (Log Double Exponential)
    "lognorm": {"p":[], "D": []},         #Log-Normal
    "lomax": {"p":[], "D": []},           #Lomax (Pareto of the second kind)
    "maxwell": {"p":[], "D": []},         #Maxwell
    "mielke": {"p":[], "D": []},          #Mielke's Beta-Kappa
    "nakagami": {"p":[], "D": []},        #Nakagami
    "ncx2": {"p":[], "D": []},            #Non-central chi-squared
    "ncf": {"p":[], "D": []},             #Non-central F
    "nct": {"p":[], "D": []},             #Non-central Student's T
    "norm": {"p":[], "D": []},            #Normal (Gaussian)
    "pareto": {"p":[], "D": []},          #Pareto
    "pearson3": {"p":[], "D": []},        #Pearson type III
    "powerlaw": {"p":[], "D": []},        #Power-function
    "powerlognorm": {"p":[], "D": []},    #Power log normal
    "powernorm": {"p":[], "D": []},       #Power normal
    "rdist": {"p":[], "D": []},           #R distribution
    "reciprocal": {"p":[], "D": []},      #Reciprocal
    "rayleigh": {"p":[], "D": []},        #Rayleigh
    "rice": {"p":[], "D": []},            #Rice
    "recipinvgauss": {"p":[], "D": []},   #Reciprocal Inverse Gaussian
    "semicircular": {"p":[], "D": []},    #Semicircular
    "t": {"p":[], "D": []},               #Student's T
    "triang": {"p":[], "D": []},          #Triangular
    "truncexpon": {"p":[], "D": []},      #Truncated Exponential
    "truncnorm": {"p":[], "D": []},       #Truncated Normal
    "tukeylambda": {"p":[], "D": []},     #Tukey-Lambda
    "uniform": {"p":[], "D": []},         #Uniform
    "vonmises": {"p":[], "D": []},        #Von-Mises (Circular)
    "wald": {"p":[], "D": []},            #Wald
    "weibull_min": {"p":[], "D": []},     #Minimum Weibull (see Frechet)
    "weibull_max": {"p":[], "D": []},     #Maximum Weibull (see Frechet)
    "wrapcauchy": {"p":[], "D": []},      #Wrapped Cauchy
    "ksone": {"p":[], "D": []},           #Kolmogorov-Smirnov one-sided (no stats)
    "kstwobign": {"p":[], "D": []}}       #Kolmogorov-Smirnov two-sided test for Large N

########################################################################################
# this part is only used to read in the file ...
# format should be :
# 0.00192904472351
# 0.0030369758606
# 0.00188088417053
# 0.00222492218018
# 0.00447607040405
# 0.00301194190979
def read(filename):
    if not options.filename == "":
        print("reading data in file %s ... " % options.filename, end="")
        f = open(options.filename)
        data = [float(value) for value in f.readlines()]
        f.close()
        print("done")

    return data

########################################################################################

def check(data, fct, verbose=False):
    #fit our data set against every probability distribution
    parameters = eval("scipy.stats."+fct+".fit(data)");
    #Applying the Kolmogorov-Smirnof two sided test
    D, p = scipy.stats.kstest(data, fct, args=parameters);

    if math.isnan(p): p=0
    if math.isnan(D): D=0

    if verbose:
        print(fct.ljust(16) + "p: " + str(p).ljust(25) + "D: " +str(D))

    return (fct, p, D)

########################################################################################

def plot(fcts, data):
    import matplotlib.pyplot as plt
    import numpy as np

    # plot data
    plt.hist(data, normed=True, bins=max(10, len(data)/10))

    # plot fitted probability
    for fct in fcts:
        params = eval("scipy.stats."+fct+".fit(data)")
        f = eval("scipy.stats."+fct+".freeze"+str(params))
        x = np.linspace(f.ppf(0.001), f.ppf(0.999), 500)
        plt.plot(x, f.pdf(x), lw=3, label=fct)
    plt.legend(loc='best', frameon=False)
    plt.title("Top "+str(len(fcts))+" Results")
    plt.show()

def plotDensities(best):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.ion()
    plt.clf()
    # plot fitted probability
    for i in range(len(best)-1, -1, -1):
        fct, values = best[i]
        plt.hist(values["p"], normed=True, bins=max(10, len(values["p"])/10), label=str(i+1)+". "+fct, alpha=0.5)
    plt.legend(loc='best', frameon=False)
    plt.title("Top Results")
    plt.show()
    plt.draw()


if __name__ == '__main__':

    #########################################################################################
    parser = OptionParser()
    parser.add_option("-f", "--file",      dest="filename",  default="",    type="string",       help="file with measurement data", metavar="FILE")
    parser.add_option("-v", "--verbose",   dest="verbose",   default=False, action="store_true", help="print all results immediately (default=False)" )
    parser.add_option("-t", "--top",       dest="top",       default=10,    type="int",          help="define amount of printed results (default=10)")
    parser.add_option("-p", "--plot",      dest="plot",      default=False, action="store_true", help="plot the best result with matplotlib (default=False)")
    parser.add_option("-i", "--iterative", dest="iterative", default=1,     type="int",          help="define number of iterative checks (default=1)")
    parser.add_option("-e", "--exclude",   dest="exclude",   default=10.0,  type="float",        help="amount (in per cent) of exluded samples for each iteration (default=10.0%)" )
    parser.add_option("-n", "--processes", dest="processes", default=-1,    type="int",          help="number of process used in parallel (default=-1...all)")
    parser.add_option("-d", "--densities", dest="densities", default=False, action="store_true", help="")
    parser.add_option("-g", "--generate",  dest="generate",  default=False, action="store_true", help="generate an example file")

    (options, args) = parser.parse_args()

    if options.generate:
        print("generating random data 'example-halflogistic.dat' ... ", end="")
        f = open("example-halflogistic.dat", "w")
        f.writelines([str(s)+"\n" for s in scipy.stats.halflogistic().rvs(500)])
        f.close()
        print("done")
        quit()

    # read data from file or generate
    DATA = read(options.filename)

    for i in range(options.iterative):

        if options.iterative == 1:
            data = DATA
        else:
            data = [value for value in DATA if random.random()>= options.exclude/100]

        results = Parallel(n_jobs=options.processes)(delayed(check)(data, fct, options.verbose) for fct in cdfs.keys())

        for res in results:
            key, p, D = res
            cdfs[key]["p"].append(p)
            cdfs[key]["D"].append(D)

        print( "-------------------------------------------------------------------" )
        print( "Top %d after %d iteration(s)" % (options.top, i+1, ) )
        print( "-------------------------------------------------------------------" )
        best = sorted(cdfs.items(), key=lambda elem : scipy.median(elem[1]["p"]), reverse=True)

        for t in range(options.top):
            fct, values = best[t]
            print( str(t+1).ljust(4), fct.ljust(16),
                   "\tp: ", scipy.median(values["p"]),
                   "\tD: ", scipy.median(values["D"]),
                   end="")
            if len(values["p"]) > 1:
                print("\tvar(p): ", scipy.var(values["p"]),
                      "\tvar(D): ", scipy.var(values["D"]), end="")
            print()

        if options.densities:
            plotDensities(best[:t+1])

    if options.plot:
        # get only the names ...
        plot([b[0] for b in best[:options.top]], DATA)
