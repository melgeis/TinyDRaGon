import urllib.request
import numpy as np
import json

import helpers

def download_elev(min_lat, min_lon, max_lat, max_lon, file: str, cell_size_m: int = 25):
	lat_dist = helpers.compute_distance(max_lat, min_lon, min_lat, min_lon)
	lon_dist = helpers.compute_distance(min_lat, max_lon, min_lat, min_lon)

	lat_steps = int(lat_dist / cell_size_m)
	lon_steps = int(lon_dist / cell_size_m)

	lat_inc = (max_lat - min_lat) / lat_steps
	lon_inc = (max_lon - min_lon) / lon_steps

	# create elevation profile matrix
	ep = np.zeros((lat_steps, lon_steps))

	# request EU-DEM data
	url = "http://topodata:5000/v1/eudem?locations="
	req = url
	ctr = 0
	cur_pos = [-1, 0]
	for i in range(lat_steps):
		_lat = min_lat + lat_inc * i

		for j in range(lon_steps):
			_lon = min_lon + lon_inc * j
			req += str(_lat) + "," + str(_lon) + "|"
			ctr += 1

			if ctr == 800: # limit is 800 locations per request
				contents = urllib.request.urlopen(req).read()
				js = json.loads(contents)

				# save elevation values in matrix
				for k in range(len(js["results"])):
					ele = js["results"][k]["elevation"]
					if ele is None:
						ele = 0

					ep[cur_pos[0]][cur_pos[1]] = ele

					# update current position
					if cur_pos[1] < lon_steps - 1:
						cur_pos[1] += 1
					else:
						cur_pos[0] -= 1
						cur_pos[1] = 0

				req = url
				ctr = 0

	if req != url:
		contents = urllib.request.urlopen(req).read()
		js = json.loads(contents)

		# save elevation values in matrix
		for k in range(-len(js["results"]), 0):
			ele = js["results"][k]["elevation"]
			if ele is None:
				ele = 0

			ep[cur_pos[0]][cur_pos[1]] = ele

			# update current position
			if cur_pos[1] < lon_steps - 1:
				cur_pos[1] += 1
			else:
				cur_pos[0] -= 1
				cur_pos[1] = 0
