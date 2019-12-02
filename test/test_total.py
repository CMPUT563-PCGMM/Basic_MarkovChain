import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import math

import numpy as np

from utils import readMaps, data_split, create_initial_room_map, four_side_bagging_sampling
from model.BasicMarkovChain import BasicMarkovChain
from model.MarkovRandomFieldModel import MarkovRandomFieldModel
from Post_Process.functions import iterate_over_map, all_confs_np, read_map_directory_linux
from PCGMM_Evaluation_Method.playability import evaluate_playability
from PCGMM_Evaluation_Method.similarity import evaluate_similarity
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

sampling_num = 2

# read rooms for trainning data (you may have to change)
TRAINING_DATA_PATH = "../PCGMM_Evaluation_Method/map_data/map_reduced_OI"

maps_data = readMaps(tileTypes, maps_path=TRAINING_DATA_PATH)
training_data, validation_data, testing_data = data_split(maps_data, train_size=0.8, validate_size=0.0, test_size=0.2)
print("Training data length = {}".format(len(training_data)))
print("Test data length = {}".format(len(testing_data)))

##############################################
'''
  Training Data
'''
print("\n"+"#"*10)
print("Training Data")

test = np.asarray(training_data, dtype=str)

param_dict = {
    "similarity_function": "histogram_base",
    "enable_cluster": True
}

# check palyability
unplayble_room, playability = evaluate_playability(test)

# check similarity
similarity_function = param_dict["similarity_function"]
enable_cluster = param_dict["enable_cluster"]

similarity = evaluate_similarity(test, similarity_function, enable_cluster)
style = len(testing_data) * math.log(1/7) * 12 * 7


report = "playability = {}\nsimilarity = {}".format(playability, similarity)
print(report)
print("test data log prob = {} on random sampling".format(style))
print("#"*10)
##############################################

##############################################
'''
  MRF
'''
print("\n"+"#"*10)
print("MRF Processing")

training_param_dict = {}
mrf = MarkovRandomFieldModel(_b=9999) 

start_time = time.time()
# print("Learning...")
mrf.learn(training_data, training_param_dict)
# print("Learning time = {}".format(time.time() - start_time))

# # Initial room map with the given border
initial_room_map = create_initial_room_map(height=16, width=11)
# print(initial_room_map)

# Learning direction: top-down-left; top-down-right; bottom-up-left; bottom-up-right
sampling_param_dict = {}

# print("Sampling...") 
rooms = np.empty((sampling_num, 16, 11), dtype = str)           
for i in range(sampling_num):
    # print(i)
    generated_map = mrf.generate_new_room(initial_room_map, sampling_param_dict)
    rooms[i, :, :] = generated_map

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
print(mrf.evaluate(rooms, evaluate_param_dict))
print("#"*10)
##############################################

##############################################
'''
  Markov Chain
'''
print("\n"+"#"*10)
print("Markov Chain Processing")
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
# print("Learning...")
basic_MC.learn(training_data, training_param_dict)
# print("Learning time = {}".format(time.time() - start_time))

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

# print("Sampling...")
rooms = np.empty((sampling_num, 16, 11), dtype = str)
for i in range(sampling_num):
    # print(i)
    initial_room_map = create_initial_room_map(height=16, width=11)
    generated_map = basic_MC.generate_new_room(initial_room_map, sampling_param_dict)
    rooms[i, :, :] = generated_map

# rooms_for_post_processing = rooms

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
print("#"*10)
##############################################


##############################################
'''
  Markov Chain + post processing
'''
print("\n"+"#"*10)
print("Markov Chain + post Processing")

dict1 = all_confs_np(training_data, True)
dict2 = all_confs_np(training_data, False)

new_rooms = np.empty((sampling_num, 16, 11), dtype = str)
for i in range(sampling_num):
    try:
        new_rooms[i, :, :] = iterate_over_map(10, 3, 3, rooms[i, :, :], dict1, dict2)
    except Exception as e:
        # print("some error")
        # print(rooms[i, :, :])
        new_rooms[i, :, :] = rooms[i, :, :]
    # print(i)
# print(new_rooms.shape)
print(basic_MC.evaluate(new_rooms, evaluate_param_dict))
print("#"*10)
##############################################

##############################################
'''
  Markov Chain + 4-side
'''
print("\n"+"#"*10)
print("Markov Chain + 4-side Processing")

rooms = np.empty((sampling_num, 16, 11), dtype = str)
for i in range(sampling_num):
    initial_room_map = create_initial_room_map(height=16, width=11)
    generated_map = four_side_bagging_sampling(basic_MC, initial_room_map, sampling_param_dict, bagging_num=3)
    rooms[i, :, :] = generated_map

print(basic_MC.evaluate(rooms, evaluate_param_dict))
print("#"*10)
##############################################

##############################################
'''
  Markov Chain + 4-side + post processing
'''
print("\n"+"#"*10)
print("Markov Chain + 4-side + post Processing")

new_rooms = np.empty((sampling_num, 16, 11), dtype = str)
for i in range(sampling_num):
    try:
        new_rooms[i, :, :] = iterate_over_map(10, 3, 3, rooms[i, :, :], dict1, dict2)
    except Exception as e:
        # print("some error")
        # print(rooms[i, :, :])
        new_rooms[i, :, :] = rooms[i, :, :]
    # print(i)
# print(new_rooms.shape)
print(basic_MC.evaluate(new_rooms, evaluate_param_dict))
print("#"*10)
##############################################


