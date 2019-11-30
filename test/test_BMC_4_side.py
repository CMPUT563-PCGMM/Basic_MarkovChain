import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np

from utils import readMaps, data_split, create_initial_room_map
from model.BasicMarkovChain import BasicMarkovChain

from eval import style_eval


def four_side_bagging_sampling(initial_room_map, sampling_param_dict, bagging_num=1):
    candidates_list = []
    # get first sampling
    for i in range(bagging_num):
        # side_1 = basic_MC.generate_new_room(initial_room_map, sampling_param_dict)
        # print(initial_room_map)
        side_1 = basic_MC.generate_new_room(initial_room_map, sampling_param_dict)
        # print(side_1)
        candidates_list.append(side_1)

        # lr flip
        # print(np.flip(initial_room_map, 1))
        side_2 = basic_MC.generate_new_room(np.flip(initial_room_map, 1), sampling_param_dict)
        # print(np.flip(side_2, 1))
        candidates_list.append(side_2)

        # ud flip
        # print(np.flip(initial_room_map, 0))
        side_3 = basic_MC.generate_new_room(np.flip(initial_room_map, 0), sampling_param_dict)
        # print(np.flip(side_3, 0))
        candidates_list.append(side_3)

        # lr && ud flip
        # print(np.flip(np.flip(initial_room_map, 0),1))
        side_4 = basic_MC.generate_new_room(np.flip(np.flip(initial_room_map, 0),1), sampling_param_dict)
        # print(np.flip(np.flip(side_4, 0), 1))
        candidates_list.append(side_4)

    
    candidates_map = np.stack(tuple(candidates_list), -1)
    # print(candidates_map.shape)
    
    # https://stackoverflow.com/questions/12297016/how-to-find-most-frequent-values-in-numpy-ndarray?rq=1
    axis = 2
    u, indices = np.unique(candidates_map, return_inverse=True)
    generated_room = u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(candidates_map.shape), None, np.max(indices) + 1), axis=axis)]
    # print(generated_room)

    return generated_room



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

# # Initial room map with the given border
# initial_room_map = create_initial_room_map(height=16, width=11)
# print(initial_room_map)

# Initial tile list, remove 'W' and 'D'
tiles = list(tileTypes.keys())

removed_tiles = ['W', 'D', 'U', 'N', 'E', 'A']

for removed_tile in removed_tiles:
    tiles.remove(removed_tile)

# Learning direction: top-down-left; top-down-right; bottom-up-left; bottom-up-right
sampling_param_dict = {'learning_direction': 'top-down-left',
                       'tiles': tiles,
                       'sampling_methods': ['lookahead', 'fallback'],
                       'lookahead': 5,
                       'fallback': True}

print("Sampling...")
sampling_num = 100
rooms = np.empty((sampling_num, 16, 11), dtype = str)
for i in range(sampling_num):
    initial_room_map = create_initial_room_map(height=16, width=11)

    generated_map = four_side_bagging_sampling(initial_room_map, sampling_param_dict, bagging_num=3)
    # generated_map = basic_MC.generate_new_room(initial_room_map, sampling_param_dict)
    rooms[i, :, :] = generated_map
    print(i)




evaluate_param_dict = {
    # for similarity
    "similarity_function": "histogram_base",
    "enable_cluster": True,
    # for style
    "check_style": True,
    "test_data": testing_data,
    "training_param_dict": training_param_dict,
    "sampling_param_dict": sampling_param_dict
}
print(basic_MC.evaluate(rooms, evaluate_param_dict))
  # np.savetxt(DETINIATION_GENERATE_ROOM+"/basic_MC_{}.txt".format(i), generated_map, fmt="%s", delimiter="")
