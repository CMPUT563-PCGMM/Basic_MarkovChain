from abc import ABC, abstractmethod

class BaseModel:

    @abstractmethod
    def learn(self, training, param_dict):
        pass

    @abstractmethod
    def generate_new_room(self, initial_room_map, param_dict):
        pass

    @abstractmethod
    def evaluate(self, evaluate_data, param_dict):
        pass