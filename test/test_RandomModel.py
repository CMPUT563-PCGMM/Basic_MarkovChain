import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from utils import create_initial_room_map
from model.RandomModel import RandomModel


# Initial room map with the given border
initial_room_map = create_initial_room_map(height=16, width=11)

sampling_param_dict = {}

rm = RandomModel()
sampling_num = 400
rooms = np.empty((sampling_num, 16, 11), dtype = str) 
for i in range(sampling_num):
    generated_map = rm.generate_new_room(initial_room_map, sampling_param_dict)
    # np.savetxt(DETINIATION_GENERATE_ROOM+"/RM_{}.txt".format(i), generated_map, fmt="%s", delimiter="") 
    rooms[i, :, :] = generated_map

evaluate_param_dict = {
  # for similarity
  "similarity_function": "histogram_base",
  "enable_cluster": True,
  # for style
  "check_style": False
}
print(rm.evaluate(rooms, evaluate_param_dict))
