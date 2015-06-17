__author__ = 'kiruba'
import ee
from ee import mapclient
from PIL import _imagingtk
ee.Initialize()

image = ee.Image("LC80420342013140LGN01")
print (image.getInfo())
mapclient.addToMap(image)