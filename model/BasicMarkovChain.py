import os
import random
import numpy as np
import math

from model.BaseModel import BaseModel
from utils import flip_map

class BasicMarkovChain(BaseModel):
    def __init__(self):
        self.dep_mat = None
        self.fallback_dep_mat = None
        # The number of times that the tile type appears (Does not include counts for 'W' and 'D')
        self.each_tile_count = dict()
        # Key: (tile_2_type, tile_1_type, tile_1_type, ...)
        # Value: the number of times that the tile type in 2 appears followed by the configuration of tile types in 1
        self.dep_mat_abs_count = dict()
        self.fallback_dep_mat_abs_count = dict()
        # Key: (tile_2_type, tile_1_type, tile_1_type, ...)
        # Value: (the number of times that the tile types in 2 appears followed by the configuration of tile types in 1) / (the number of times that the configuration of tile types in 1 appears)
        self.dep_mat_cond_prob = dict()
        self.fallback_dep_mat_cond_prob = dict()

    def learn(self, training_data, param_dict):

        def map_single_tile_count(map_data):
            map_h = map_data.shape[0]
            map_w = map_data.shape[1]

            for row_i in range(map_h):
                for col_j in range(map_w):
                    if map_data[row_i, col_j] != 'W' and map_data[row_i, col_j] != 'D':
                        if map_data[row_i, col_j] not in self.each_tile_count:
                            self.each_tile_count[map_data[row_i, col_j]] = 1
                        else:
                            self.each_tile_count[map_data[row_i, col_j]] += 1

        def map_abs_count(map_data, dependency_matrix, is_fallback_dep_mat):
            map_h = map_data.shape[0]
            map_w = map_data.shape[1]

            dep_h = dependency_matrix.shape[0]
            dep_w = dependency_matrix.shape[1]

            for row_i in range(0, map_h-dep_h+1):
                for col_j in range(0, map_w-dep_w+1):
                    sub_map = map_data[row_i:row_i+dep_h, col_j:col_j+dep_w]

                    tile_2_mark = dependency_matrix == 2
                    tile_1_mark = dependency_matrix == 1

                    tile_2_val = None
                    tile_1s_val = []

                    for sub_r_i in range(dep_h):
                        for sub_c_j in range(dep_w):
                            if dependency_matrix[sub_r_i][sub_c_j] == 1:
                                tile_1s_val.append(sub_map[sub_r_i][sub_c_j])

                            if dependency_matrix[sub_r_i][sub_c_j] == 2:
                                tile_2_val = sub_map[sub_r_i][sub_c_j]

                    tile_config = tuple([tile_2_val] + tile_1s_val)

                    if is_fallback_dep_mat:
                        if tile_config not in self.fallback_dep_mat_abs_count:
                            self.fallback_dep_mat_abs_count[tile_config] = 1
                        else:
                            self.fallback_dep_mat_abs_count[tile_config] += 1
                    else:
                        if tile_config not in self.dep_mat_abs_count:
                            self.dep_mat_abs_count[tile_config] = 1
                        else:
                            self.dep_mat_abs_count[tile_config] += 1

            return

        def cond_prob(abs_count_dict):
            cond_prob_dict = dict()

            # Key: the configuration of tile types in 1
            # Value: the number of times that the configuration of tile types in 1 appears
            tile1s_config_dict = dict()

            for tiles_config in abs_count_dict:
                tile1s_config = tiles_config[1:]

                if tile1s_config not in tile1s_config_dict:
                    tile1s_config_dict[tile1s_config] = abs_count_dict[tiles_config]
                else:
                    tile1s_config_dict[tile1s_config] += abs_count_dict[tiles_config]

            for tiles_config in abs_count_dict:
                cond_prob_dict[tiles_config] = abs_count_dict[tiles_config] / tile1s_config_dict[tiles_config[1:]]

            return cond_prob_dict


        self.dep_mat = param_dict['dep_mat']
        self.fallback_dep_mat = param_dict['fallback_dep_mat']

        for map in training_data:
            # Check for the learning direction
            if param_dict['learning_direction'] == 'top-down-left':
                # No need to flip the map
                pass

            if param_dict['learning_direction'] == 'top-down-right':
                # Flip the map hortizontally
                map = flip_map(map, True, False)

            if param_dict['learning_direction'] == 'bottom-up-left':
                # Flip the map vertically
                map = flip_map(map, False, True)

            if param_dict['learning_direction'] == 'bottom-up-right':
                # Flip the map horizontally and vertically
                map = flip_map(map, True, True)

            map_single_tile_count(map)
            map_abs_count(map, self.dep_mat, False)
            map_abs_count(map, self.fallback_dep_mat, True)

        self.dep_mat_cond_prob = cond_prob(self.dep_mat_abs_count)
        self.fallback_dep_mat_cond_prob = cond_prob(self.fallback_dep_mat_abs_count)

    def generate_new_room(self, initial_room_map, param_dict):

        def get_the_next_loc(map, row_idx, col_idx):
            found_current = False

            map_h = map.shape[0]
            map_w = map.shape[1]

            for row_i in range(2, map_h-2):
                for col_j in range(2, map_w-2):
                    if found_current:
                        return row_i, col_j

                    if row_i == row_idx and col_j == col_idx:
                        found_current = True
            # The next location is out of map
            return -1, -1

        def generate_tile_from_p(cond_prob_dict, config, avail_tiles):
            # Generate the tile from cond_prob_dict given c and avail_tiles
            cond_prob_given_c = []
            corresp_tile2 = []

            for key in cond_prob_dict:
                if list(key[1:]) == config and key[0] in avail_tiles:
                    cond_prob_given_c.append(cond_prob_dict[key])
                    corresp_tile2.append(key[0])

            # Normalize cond_prob_given_c
            normed_cond_prob_given_c = []

            for prob_idx in range(len(cond_prob_given_c)):
                normed_cond_prob_given_c.append(cond_prob_given_c[prob_idx] / sum(cond_prob_given_c))

            generated_t = None

            if len(corresp_tile2) > 0:
                sampled_idx = np.random.choice(np.arange(len(corresp_tile2)), p=normed_cond_prob_given_c)
                generated_t = corresp_tile2[sampled_idx]

            return generated_t

        def sample_tile(map, row_idx, col_idx, lookahead_num, dependency_matrix, cond_prob_dict, tiles_lst):
            if lookahead_num < 0 or (row_idx == -1 or col_idx == -1):
                return True, map

            avail_tiles_lst = tiles_lst
            current_map = map

            # current config of the map, (row_idx, col_idx) is the location of the tile 2 in the dependency_matrix
            tile2_row_idx = min(np.where(np.any(dependency_matrix==2, axis=1))[0])
            tile2_col_idx = min(np.where(np.any(dependency_matrix==2, axis=0))[0])

            # Find the orig of dep_mat when generate this tile
            orig_row_i = row_i - tile2_row_idx
            orig_col_j = col_j - tile2_col_idx

            dep_h = dependency_matrix.shape[0]
            dep_w = dependency_matrix.shape[1]

            sub_map = current_map[orig_row_i:orig_row_i+dep_h, orig_col_j:orig_col_j+dep_w]

            # The config for this sub_map
            config = []

            for sub_r_i in range(dep_h):
                for sub_c_j in range(dep_w):
                    if dependency_matrix[sub_r_i][sub_c_j] == 1:
                        config.append(sub_map[sub_r_i][sub_c_j])

            # Check if the config is unseen
            is_unseen = True

            for seen_state in list(cond_prob_dict.keys()):
                if list(seen_state[1:]) == config:
                    is_unseen = False
                    break

            if is_unseen:
                return False, current_map
            else:
                # Sampling a tile given the current config
                generated_t = generate_tile_from_p(cond_prob_dict, config, avail_tiles_lst)

                if generated_t is not None:
                    current_map[row_idx, col_idx] = generated_t
                else:
                    return False, current_map

                while True:
                    nex_row_idx, nex_col_idx = get_the_next_loc(current_map, row_idx, col_idx)

                    sample_tile_res, nex_current_map = sample_tile(current_map, nex_row_idx, nex_col_idx, lookahead_num-1, dependency_matrix, cond_prob_dict, tiles_lst)

                    if not sample_tile_res:
                        if len(avail_tiles_lst) > 0:
                            avail_tiles_lst.remove(generated_t)
                        else:
                            return False, current_map

                        if len(avail_tiles_lst) == 0:
                            return False, current_map
                        else:
                            new_generated_t = generate_tile_from_p(cond_prob_dict, config, avail_tiles_lst)

                            current_map[row_idx, col_idx] = new_generated_t
                    else:
                        break

                return sample_tile_res, current_map


        # Check for the learning direction
        if param_dict['learning_direction'] == 'top-down-left':
            # No need to flip the map
            pass

        if param_dict['learning_direction'] == 'top-down-right':
            # Flip the map hortizontally
            initial_room_map = flip_map(initial_room_map, True, False)

        if param_dict['learning_direction'] == 'bottom-up-left':
            # Flip the map vertically
            initial_room_map = flip_map(initial_room_map, False, True)

        if param_dict['learning_direction'] == 'bottom-up-right':
            # Flip the map horizontally and vertically
            initial_room_map = flip_map(initial_room_map, True, True)

        initial_room_map_h = initial_room_map.shape[0]
        initial_room_map_w = initial_room_map.shape[1]

        generated_map = initial_room_map

        # The number of randomly generated tiles
        rand_tile_num = 0

        # Iterate through the sampling-methods
        lookahead_num_param = 0
        fallback_param = False

        for sampling_method in param_dict['sampling_methods']:
            if sampling_method == 'lookahead':
                lookahead_num_param = param_dict['lookahead']

            if sampling_method == 'fallback':
                fallback_param = param_dict['fallback']

        all_tiles_lst = param_dict['tiles'].copy()

        for row_i in range(2, initial_room_map_h-2):
            for col_j in range(2, initial_room_map_w-2):

                this_tiles_lst = all_tiles_lst.copy()

                if fallback_param:
                    dep_mat_lst = [self.dep_mat, self.fallback_dep_mat]
                    cond_prob_dict_lst = [self.dep_mat_cond_prob, self.fallback_dep_mat_cond_prob]
                else:
                    dep_mat_lst = [self.dep_mat]
                    cond_prob_dict_lst = [self.dep_mat_cond_prob]

                prev_generated_map = generated_map

                for dep_mat_idx in range(len(dep_mat_lst)):
                    sample_tile_res, generated_map = sample_tile(prev_generated_map, row_i, col_j, lookahead_num_param, dep_mat_lst[dep_mat_idx], cond_prob_dict_lst[dep_mat_idx], this_tiles_lst)

                    if sample_tile_res:
                        # None need to generate the tile using the fallback dependency matrix
                        break

                if not sample_tile_res:
                    # sample a tile failed, generate a tile randomly
                    tile_prob = []
                    corresp_tile = []

                    for key in self.each_tile_count:
                        tile_prob.append(self.each_tile_count[key] / sum(self.each_tile_count.values()))
                        corresp_tile.append(key)

                    sampled_idx = np.random.choice(np.arange(len(corresp_tile)), p=tile_prob)
                    generated_t = corresp_tile[sampled_idx]

                    prev_generated_map[row_i, col_j] = generated_t

                    generated_map = prev_generated_map

                    rand_tile_num += 1

        print("rand_tile_num: ", rand_tile_num)

        # Flip the generated map corresponding to the learning direction
        if param_dict['learning_direction'] == 'top-down-right':
            # Flip the map hortizontally
            generated_map = flip_map(generated_map, True, False)

        if param_dict['learning_direction'] == 'bottom-up-left':
            # Flip the map vertically
            generated_map = flip_map(generated_map, False, True)

        if param_dict['learning_direction'] == 'bottom-up-right':
            # Flip the map horizontally and vertically
            generated_map = flip_map(generated_map, True, True)

        return generated_map

    def style_evaluate(self, test_data, training_param_dict, sampling_param_dict):
        '''
        trained_model: trained model
        test_data: list of generated images
        param_dict: the param_dict used for training the model
        '''

        logP_img_sum = 0.0

        for generated_img in test_data:
            # P(generated_img | trained_model) = multiplication of P(tile2 | tile1s) for every gerenrated tiles
            # log(P(generated_img | trained_model)) = sum of log(P(tile2 | tile1s)) for evert generated tiles
            logP_img = 0.0

            # Check for the learning direction
            if training_param_dict['learning_direction'] == 'top-down-left':
                # No need to flip the map
                gen_img = generated_img

            if training_param_dict['learning_direction'] == 'top-down-right':
                # Flip the map hortizontally
                gen_img = flip_map(generated_img, True, False)

            if training_param_dict['learning_direction'] == 'bottom-up-left':
                # Flip the map vertically
                gen_img = flip_map(generated_img, False, True)

            if training_param_dict['learning_direction'] == 'bottom-up-right':
                # Flip the map horizontally and vertically
                gen_img = flip_map(generated_img, True, True)

            gen_img_h = gen_img.shape[0]
            gen_img_w = gen_img.shape[1]

            dependency_matrices = [training_param_dict['dep_mat']]

            if sampling_param_dict['fallback']:
                dependency_matrices.append(training_param_dict['fallback_dep_mat'])


            for row_i in range(2, gen_img_h-2):
                for col_j in range(2, gen_img_w-2):
                    # Get the sub map of the dependency matrix with tile 2 at (row_i, col_j)

                    for dep_mat_idx in range(len(dependency_matrices)):
                        # current config of the map, (row_idx, col_idx) is the location of the tile 2 in the dependency_matrix
                        tile2_row_idx = min(np.where(np.any(dependency_matrices[dep_mat_idx]==2, axis=1))[0])
                        tile2_col_idx = min(np.where(np.any(dependency_matrices[dep_mat_idx]==2, axis=0))[0])

                        # Find the orig of dep_mat when generate this tile
                        orig_row_i = row_i - tile2_row_idx
                        orig_col_j = col_j - tile2_col_idx

                        dep_h = dependency_matrices[dep_mat_idx].shape[0]
                        dep_w = dependency_matrices[dep_mat_idx].shape[1]

                        sub_map = generated_img[orig_row_i:orig_row_i+dep_h, orig_col_j:orig_col_j+dep_w]

                        # The config for this sub_map
                        config = []
                        # The generated tile 2 value
                        tile2_val = None

                        for sub_r_i in range(dep_h):
                            for sub_c_j in range(dep_w):
                                if dependency_matrices[dep_mat_idx][sub_r_i][sub_c_j] == 1:
                                    config.append(sub_map[sub_r_i][sub_c_j])

                                if dependency_matrices[dep_mat_idx][sub_r_i][sub_c_j] == 2:
                                    tile2_val = [sub_map[sub_r_i][sub_c_j]]

                        # The key for the cond_prob in the trained model
                        key_val = tuple(tile2_val + config)

                        if dep_mat_idx == 0:
                            # Using the trained_model.dep_mat_cond_prob
                            if key_val in self.dep_mat_cond_prob and self.dep_mat_cond_prob[key_val] != 0:
                                logP_img += math.log(self.dep_mat_cond_prob[key_val])

                                # Found this tile 2 cond prob in dep_mat_cond_prob, break and check the next generated tile
                                break

                        if dep_mat_idx == 1:
                            # Using the trained_model.fallback_dep_mat_cond_prob
                            if key_val in self.fallback_dep_mat_cond_prob and self.fallback_dep_mat_cond_prob[key_val] != 0:
                                logP_img += math.log(self.fallback_dep_mat_cond_prob[key_val])

                                # Found this tile2 cond prob in fallback_dep_mat_cond_prob, break and check the next generated tile
                                break

                        # If this tile 2 cond prob is not in both dep_mat_cond_prob and fallback_dep_mat_cond_prob, add nothing to logP_img, and check the next

            # summed all log(P(tile2 | tile1s)) for evert generated tiles for this generated image
            logP_img_sum += logP_img

        # print(logP_img_sum)
        
        return logP_img_sum
