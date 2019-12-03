import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np

from utils import readMaps, data_split, create_initial_room_map

from model.Multilayer_MC import Multilayer_MC
from Post_Process.functions import iterate_over_map, all_confs_np, read_map_directory_linux

from eval import style_eval

tileTypes = {
    "F": "FLOOR",
    "B": "BLOCK",
    "M": "MONSTER",
    "P": "ELEMENT (LAVA, WATER)",
    "D": "DOOR",
    "S": "STAIR",
    "W": "WALL",
    "-": "VOID",
    "U": "single arrow, out - Go out of the room",
    "N": "single arrow, in - Go in to the room",
    "E": "double arrow - Go in and out of the room",
    "C": "Movable block",
    "A": "Breakable wall"
}

# read rooms for trainning data (you may have to change)
TRAINING_DATA_PATH = "../PCGMM_Evaluation_Method/map_data/map_reduced_OI"

maps_data = readMaps(tileTypes, maps_path=TRAINING_DATA_PATH)
training_data, validation_data, testing_data = data_split(maps_data, train_size=0.8, validate_size=0.0, test_size=0.2)
# training_data = maps_data


training_param_dict = {}

multi_mc = Multilayer_MC()

start_time = time.time()
print("Learning...")
multi_mc.learn(training_data, training_param_dict)
print("Learning time = {}".format(time.time() - start_time))


# Learning direction: top-down-left; top-down-right; bottom-up-left; bottom-up-right
sampling_param_dict = {}

print("Sampling...")
sampling_num = 100
rooms = np.empty((sampling_num, 16, 11), dtype = str)
for i in range(sampling_num):
    # print(i)
    initial_room_map = create_initial_room_map(height=16, width=11)
    generated_map = multi_mc.generate_new_room(initial_room_map, sampling_param_dict)
    rooms[i, :, :] = generated_map
    # print(generated_map)
    # print(generated_map.shape)


print("get 3*3 conf ...")
dict1 = all_confs_np(training_data, True)
dict2 = all_confs_np(training_data, False)
print("finished")

# print(rooms.shape)
new_rooms = np.empty((sampling_num, 16, 11), dtype = str)
for i in range(sampling_num):
    try:
        new_rooms[i, :, :] = iterate_over_map(10, 3, 3, np.fliplr(rooms[i, :, :]), dict1, dict2)
    except Exception as e:
        print("some error")
        print(rooms[i, :, :])
        new_rooms[i, :, :] = rooms[i, :, :]
    print(i)



evaluate_param_dict = {
  # for similarity
  "similarity_function": "histogram_base",
  "enable_cluster": True,
  # for style
  "check_style": False,
}
print(multi_mc.evaluate(new_rooms, evaluate_param_dict))
  # np.savetxt(DETINIATION_GENERATE_ROOM+"/basic_MC_{}.txt".format(i), generated_map, fmt="%s", delimiter="")
