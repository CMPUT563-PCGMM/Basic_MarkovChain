import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np

from utils import readMaps, data_split, create_initial_room_map
from model.MarkovRandomFieldModel import MarkovRandomFieldModel

tileTypes = {
    "F": "FLOOR",
    "B": "BLOCK",
    "M": "MONSTER",
    "P": "ELEMENT (LAVA, WATER)",
    "D": "DOOR",
    "S": "STAIR",
    "W": "WALL",
    "-": "VOID",
    "A": "BREAKABLE WALL",
    "C": "MOVABLE BLOCK",
    "N": "TODO",
    "E": "TODO",
    "U": "TODO"
}

# read rooms for trainning data (you may have to change)
TRAINING_DATA_PATH = "../PCGMM_Evaluation_Method/map_data/map_reduced_OI"

maps_data = readMaps(tileTypes, maps_path=TRAINING_DATA_PATH)
# maps_data = readMaps(tileTypes, maps_path="./maps")
# training_data, validation_data, testing_data = data_split(maps_data)
training_data = maps_data

training_param_dict = {}

mrf = MarkovRandomFieldModel(_b=999) 

start_time = time.time()
print("Learning...")
mrf.learn(training_data, training_param_dict)
print("Learning time = {}".format(time.time() - start_time))

# # Initial room map with the given border
initial_room_map = create_initial_room_map(height=16, width=11)
# print(initial_room_map)

# Learning direction: top-down-left; top-down-right; bottom-up-left; bottom-up-right
sampling_param_dict = {}

print("Sampling...") 
sampling_num = 400
rooms = np.empty((sampling_num, 16, 11), dtype = str)           
for i in range(sampling_num):
    generated_map = mrf.generate_new_room(initial_room_map, sampling_param_dict)
    rooms[i, :, :] = generated_map

print(mrf.evaluate(rooms, 
                    {"similarity_function": "histogram_base",
                    "enable_cluster": True}))