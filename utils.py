import os

import math
import random
import numpy as np


def readMaps(tileTypes, maps_path):
    maps_lst = []

    for fileName in os.listdir(maps_path):
        # print(fileName)
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
    validation_data = maps_data[int(train_size*len(maps_data)):int(train_size*len(maps_data))+int(validate_size*len(maps_data))]
    testing_data = maps_data[int(train_size*len(maps_data))+int(test_size*len(maps_data)):]

    return training_data, validation_data, testing_data

def flip_map(map, flip_hor, flip_ver):
    if flip_hor:
        # Flip the map horizontally
        map = np.fliplr(map)

    if flip_ver:
        # Flip the map vertically
        map = np.flipud(map)

    return map

def create_initial_room_map(height=10, width=8):
    initial_room = np.empty(shape=(height, width), dtype=str)
    # initial_room = np.empty(shape=(16, 11), dtype=str)

    for row_i in range(initial_room.shape[0]):
        if row_i == 0 or row_i == 1 or row_i == initial_room.shape[0] - 2 or row_i == initial_room.shape[0] - 1:
            # Fill with the wall tile: 'W'
            initial_room[row_i, :] = 'W'
        else:
            initial_room[row_i, 0:2] = 'W'
            initial_room[row_i, -2:] = 'W'

    # Assign the door tiles
    # initial_room[4:6, 1] = 'D'
    # initial_room[4:6, -2] = 'D'
    # initial_room[-2, 3:5] = 'D'
    half_height = math.floor(height / 2)
    half_width = math.floor(width / 2)
    initial_room[half_height:half_height+2, 1] = 'D'
    initial_room[half_height:half_height+2, -2] = 'D'
    initial_room[-2, half_width:half_width+2] = 'D'


    return initial_room