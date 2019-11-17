from BaseModel import BaseModel
import random
import os

from PCGMM_Evaluation_Method.playability import evaluate_playability
from PCGMM_Evaluation_Method.similarity import evaluate_similarity

class RandomModel(BaseModel):

    def __init__(self):
        # self.tile = ["F", "B", "M", "P", "O", "I", "S", "-"]
        self.tile = ["F", "B", "M", "P", "S", "-", "C"]

    def learn(self, training_data, param_dict):
        pass

    def generate_new_room(self, initial_room_map, param_dict):
        # print(initial_room_map)

        empty_field = initial_room_map[2:-2, 2:-2]
        # print(empty_field)

        for i in range(empty_field.shape[0]):
            for j in range(empty_field.shape[1]):
                empty_field[i, j] = random.choice(self.tile)

        return initial_room_map

    def evaluate(self, evaluate_data, param_dict):
        dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"
        if not os.path.isdir(dir_path+'PCGMM_Evaluation_Method'):
            raise Exception("Please clone PCGMM_Evaluation_Method under current dir")
        
        print("start evaluating ..")
        # check palyability
        unplayble_room, playability = evaluate_playability(evaluate_data)

        # check similarity
        similarity_function = param_dict["similarity_function"]
        enable_cluster = param_dict["enable_cluster"]
        similarity = evaluate_similarity(evaluate_data, similarity_function, enable_cluster)

        report = "playability = {}\nsimilarity = {}".format(playability, similarity)

        return report