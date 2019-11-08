import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np

from utils import readMaps, data_split, create_initial_room_map
from BasicMarkovChain import BasicMarkovChain

tileTypes = {
    "F": "FLOOR",
    "B": "BLOCK",
    "M": "MONSTER",
    "P": "ELEMENT (LAVA, WATER)",
    "O": "ELEMENT + FLOOR (LAVA/BLOCK, WATER/BLOCK)",
    "I": "ELEMENT + BLOCK",
    "D": "DOOR",
    "S": "STAIR",
    "W": "WALL",
    "-": "VOID"
}

maps_data = readMaps(tileTypes, maps_path="./maps")

training_data, validation_data, testing_data = data_split(maps_data)

# top-down-left dependency matrices
D_1 = np.asarray([[1, 2]], dtype=np.int8)
D_2 = np.asarray([[0, 1], [1, 2]], dtype=np.int8)
D_3 = np.asarray([[1, 1, 2]], dtype=np.int8)
D_4 = np.asarray([[1, 0], [1, 2]], dtype=np.int8)
D_5 = np.asarray([[1, 1], [1, 2]], dtype=np.int8)
D_6 = np.asarray([[0, 0, 1], [1, 1, 2]], dtype=np.int8)

training_param_dict = {'dep_mat': D_5,
                       'fallback_dep_mat': D_1,
                       'learning_direction': 'top-down-left'}

basic_MC = BasicMarkovChain()

start_time = time.time()
print("Learning...")
basic_MC.learn(training_data, training_param_dict)
print("Learning time = {}".format(time.time() - start_time))

# Initial room map with the given border
initial_room_map = create_initial_room_map(height=16, width=11)

# Initial tile list, remove 'W' and 'D'
tiles = list(tileTypes.keys())
tiles.remove('W')
tiles.remove('D')

# Learning direction: top-down-left; top-down-right; bottom-up-left; bottom-up-right
sampling_param_dict = {'learning_direction': 'top-down-left',
                       'tiles': tiles,
                       'sampling_methods': ['lookahead', 'fallback'],
                       'lookahead': 5,
                       'fallback': True}

print("Sampling...") 
# DETINIATION_GENERATE_ROOM = "./generate_map_BMC"  
DETINIATION_GENERATE_ROOM = "../../PCGMM_Evaluation_Method/generate_map_BMC"                          
for i in range(400):
  generated_map = basic_MC.generate_new_room(initial_room_map, sampling_param_dict)
  np.savetxt(DETINIATION_GENERATE_ROOM+"/basic_MC_{}.txt".format(i), generated_map, fmt="%s", delimiter="")

# print(generated_map)
