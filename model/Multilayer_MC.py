from model.BaseModel import BaseModel

from multilayer.function_multi_layer import *

class Multilayer_MC(BaseModel):

    def __init__(self):
        self.list_of_dependency = []
        self.list_of_models = []

    def learn(self, training, param_dict):
        self.regions_map = code_map_regions(16, 11)
        self.list_of_dependency = generate_dependency_matrix()
        self.list_of_models = create_all_models(self.list_of_dependency, self.regions_map, training)

    def generate_new_room(self, initial_room_map, param_dict):
        x = np.random.choice(2,4)
        while sum(x)==0:
            x = np.random.choice(2,4)
        generated_map = generat_the_map(initial_room_map, self.regions_map, self.list_of_models, self.list_of_dependency)

        return generated_map