import os

import math
import random
import numpy as np


def readMaps(tileTypes, maps_path):
    # maps_path = "./maps"

    maps_lst = []

    for fileName in os.listdir(maps_path):
        if fileName.split(".")[-1] != "txt":
            continue
        
        map = []
        # Read this map
        map_f = open(maps_path+"/"+fileName, 'r')
        # print(fileName)
        for row in map_f:
            row_chars = []
            for char in row.rstrip():
                if char not in tileTypes:
                    print('Invalid char')
                    print(char)
                row_chars.append(char)
            map.append(row_chars)

        map_arr = np.asarray(map, dtype=str)
        maps_lst.append(map_arr)

    return maps_lst

def data_split(maps_data, train_size=0.8, validate_size=0.1, test_size=0.1):
    # 80% training, 10% validation, 10% testing
    random.shuffle(maps_data)

    training_data = maps_data[:int(train_size*len(maps_data))]
    # validation_data = maps_data[int(train_size*len(maps_data)):int(train_size*len(maps_data))+int(validate_size*len(maps_data))]
    # testing_data = maps_data[int(train_size*len(maps_data))+int(test_size*len(maps_data)):-1]
    testing_data = maps_data[int(train_size*len(maps_data)):]

    # return training_data, validation_data, testing_data
    return training_data, [], testing_data

def flip_map(map, flip_hor, flip_ver):
    if flip_hor:
        # Flip the map horizontally
        map = np.fliplr(map)

    if flip_ver:
        # Flip the map vertically
        map = np.flipud(map)

    return map

def create_initial_room_map(height, width):
    door_tiles = ['D', 'U', 'N', 'E', 'A']

    initial_room = np.empty(shape=(height, width), dtype=str)

    for row_i in range(initial_room.shape[0]):
        if row_i == 0 or row_i == 1 or row_i == initial_room.shape[0] - 1 or row_i == initial_room.shape[0] - 2:
            initial_room[row_i, :] = 'W'
        else:
            initial_room[row_i, 0:2] = 'W'
            initial_room[row_i, -2:] = 'W'

    door_arr = np.zeros((4, ), dtype = np.int8)

    for door_i in range(door_arr.shape[0]):
        door_arr[door_i] = random.randint(0,1)

    if not np.any(door_arr == 1):
        # Assign a door to any side
        side_idx = random.randint(0,3)

        door_arr[side_idx] = 1

    if sum(door_arr == 1) == 1:
        # Remove U, N from door tiles
        door_tiles.remove('U')
        door_tiles.remove('N')

    if door_arr[0] == 1:
        # Top side has the one of the door tiles
        door_tile_type_idx = random.randint(0,len(door_tiles)-1)

        start_idx = width // 2

        initial_room[1, start_idx-1:start_idx+2] = door_tiles[door_tile_type_idx]

    if door_arr[1] == 1:
        # Right side has the one of the door tiles
        door_tile_type_idx = random.randint(0,len(door_tiles)-1)

        start_idx = height // 2

        initial_room[start_idx-1:start_idx+1, width-2] = door_tiles[door_tile_type_idx]

    if door_arr[2] == 1:
        # Bottom side has the one of the door tiles
        door_tile_type_idx = random.randint(0,len(door_tiles)-1)

        start_idx = width // 2

        initial_room[height-2, start_idx-1:start_idx+2] = door_tiles[door_tile_type_idx]

    if door_arr[3] == 1:
        # Left side has the one of the door tiles
        door_tile_type_idx = random.randint(0,len(door_tiles)-1)

        start_idx = height // 2

        initial_room[start_idx-1:start_idx+1, 1] = door_tiles[door_tile_type_idx]

    return initial_room

def four_side_bagging_sampling(model, initial_room_map, sampling_param_dict, bagging_num=1):
    candidates_list = []
    # get first sampling
    for i in range(bagging_num):
        # side_1 = model.generate_new_room(initial_room_map, sampling_param_dict)
        # print(initial_room_map)
        side_1 = model.generate_new_room(initial_room_map, sampling_param_dict)
        # print(side_1)
        candidates_list.append(side_1)

        # lr flip
        # print(np.flip(initial_room_map, 1))
        side_2 = model.generate_new_room(np.flip(initial_room_map, 1), sampling_param_dict)
        # print(np.flip(side_2, 1))
        candidates_list.append(side_2)

        # ud flip
        # print(np.flip(initial_room_map, 0))
        side_3 = model.generate_new_room(np.flip(initial_room_map, 0), sampling_param_dict)
        # print(np.flip(side_3, 0))
        candidates_list.append(side_3)

        # lr && ud flip
        # print(np.flip(np.flip(initial_room_map, 0),1))
        side_4 = model.generate_new_room(np.flip(np.flip(initial_room_map, 0),1), sampling_param_dict)
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
