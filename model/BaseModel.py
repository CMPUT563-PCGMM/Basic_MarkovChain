# from abc import ABC, abstractmethod
import os

from PCGMM_Evaluation_Method.playability import evaluate_playability
from PCGMM_Evaluation_Method.similarity import evaluate_similarity

class BaseModel:

    # @abstractmethod
    def learn(self, training, param_dict):
        pass

    # @abstractmethod
    def generate_new_room(self, initial_room_map, param_dict):
        pass

    # @abstractmethod
    def evaluate(self, evaluate_data, param_dict):
        dir_path = os.path.dirname(os.path.realpath(__file__)) + "/../"
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