__author__ = 'kiruba'
# import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
# import itertools
import lxml.html, urllib2, urlparse


# this code is based on http://stackoverflow.com/a/6223422/2632856

base_url = "http://iahs.info/Publications-News.do?dmsSearch_pubno=338#iahs-dms-tabs-content"

# fetch page
res = urllib2.urlopen(base_url)

# parse the response into an xml tree
tree = lxml.html.fromstring(res.read())

# construct a namespace dictionary to pass to the xpath() call
# this lets us use regular expressions in the xpath

ns = {'re': 'http://exslt.org/regular-expressions'}

# iterate over all <a> tags whose href ends in ".pdf" (case-insensitive)

for node in tree.xpath('//a[re:test(@href, "\.pdf$", "i")]', namespaces=ns):
    # print the href, joining it to the base_url
    url =  urlparse.urljoin(base_url, node.attrib['href'])
    with open("/media/kiruba/New Volume/Documents/hydrology/Hydrocomplexity_new_tools_for_solving_wicked_water_problems/download_url.txt", "a") as text_file:
        text_file.write(url.encode('utf-8') + "\n")
