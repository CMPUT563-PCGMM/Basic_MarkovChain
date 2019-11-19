import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np

from utils import readMaps, data_split, create_initial_room_map
from PCGMM_Evaluation_Method.playability import evaluate_playability
from PCGMM_Evaluation_Method.similarity import evaluate_similarity

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
maps_data = np.asarray(maps_data, dtype=str)
print(maps_data.shape)

param_dict = {
    "similarity_function": "histogram_base",
    "enable_cluster": True
}

# check palyability
unplayble_room, playability = evaluate_playability(maps_data)

# check similarity
similarity_function = param_dict["similarity_function"]
enable_cluster = param_dict["enable_cluster"]
similarity = evaluate_similarity(maps_data, similarity_function, enable_cluster)

report = "playability = {}\nsimilarity = {}".format(playability, similarity)
print(report)
