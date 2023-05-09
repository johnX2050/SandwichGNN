import numpy as np
import pandas as pd

loc_filename = 'graph_sensor_locations.csv'
df_loc = pd.read_csv(loc_filename)
np_loc = df_loc.to_numpy()
np_loc = np_loc[:, -2:]

np.savez('loc.npz', np_loc)

