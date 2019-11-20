import numpy as np
import random
import math

def style_eval(trained_model, test_data, training_param_dict, sampling_param_dict):
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
                        if key_val in trained_model.dep_mat_cond_prob and trained_model.dep_mat_cond_prob[key_val] != 0:
                            logP_img += math.log(trained_model.dep_mat_cond_prob[key_val])

                            # Found this tile 2 cond prob in dep_mat_cond_prob, break and check the next generated tile
                            break

                    if dep_mat_idx == 1:
                        # Using the trained_model.fallback_dep_mat_cond_prob
                        if key_val in trained_model.fallback_dep_mat_cond_prob and trained_model.fallback_dep_mat_cond_prob[key_val] != 0:
                            logP_img += math.log(trained_model.fallback_dep_mat_cond_prob[key_val])

                            # Found this tile2 cond prob in fallback_dep_mat_cond_prob, break and check the next generated tile
                            break

                    # If this tile 2 cond prob is not in both dep_mat_cond_prob and fallback_dep_mat_cond_prob, add nothing to logP_img, and check the next

        # summed all log(P(tile2 | tile1s)) for evert generated tiles for this generated image
        logP_img_sum += logP_img

    print(logP_img_sum)

    return logP_img_sum
