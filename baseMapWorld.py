from mpl_toolkits.basemap import Basemap
import Initialization as init
import matplotlib.pyplot as plt
import numpy as np


profiles = init.getOriProfiles()

hour1Diagram = np.array([]) 
hour2Diagram = np.array([]) 
hour3Diagram = np.array([])

latitude = np.array([])
longitude = np.array([])

for item in profiles :
	hour1 = profiles[item][0]
	hour2 = profiles[item][1]
	hour3 = profiles[item][2]

	if hour1 != 25 : hour1Diagram = np.append(hour1Diagram, hour1)
	if hour2 != 25 : hour2Diagram = np.append(hour2Diagram, hour2)
	if hour3 != 25 : hour3Diagram = np.append(hour3Diagram, hour3)

	latitude = np.append(latitude, profiles[item][3] )
	longitude = np.append(longitude, profiles[item][4] )
 
# make sure the value of resolution is a lowercase L,
#  for 'low', not a numeral 1 parms : eck4, cyl

map = Basemap(projection='cyl', lat_0=0, lon_0=0,
              resolution='l', area_thresh=1000.0)
map.scatter(longitude, latitude, 1.2, color = 'red')
 
map.drawcoastlines()
map.drawcountries()
#map.fillcontinents(color='red')
map.drawmapboundary()
 
map.drawmeridians(np.arange(0, 360, 30))
map.drawparallels(np.arange(-90, 90, 30))
 
plt.show()