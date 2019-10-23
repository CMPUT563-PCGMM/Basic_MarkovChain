import os
import numpy as np



def readMaps(maps_path, tileTypes):
    maps_lst = []

    for fileName in os.listdir(maps_path):
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
        # break

    return maps_lst



class basicMarkovChain:
    def __init__(self, maps_lst, dep_matrices, learning_dr, dep_mat_idx):
        self.maps_lst = maps_lst
        self.dep_matrices = dep_matrices
        self.learning_dr = learning_dr

        # key: each encoutered combination of s_i, s_i-1, ..., s_i-k; value: number of times
        self.abs_counts = dict()
        self.each_tile_abs_counts = dict()
        self.cond_prob_distr = dict()
        self.each_tile_prob_distr = dict()

        # Absolute counts and CPDs for each dependency matrix
        for D_idx in range(len(self.dep_matrices)):
            self.abs_counts[D_idx] = dict()
            self.cond_prob_distr[D_idx] = dict()

    def absolute_Counts(self):
        # k is the number of 1s in the dependency matrix dep_mat, s_i is 2 in dep_mat
        # Count T(s_i | s_i-1, ..., s_i-k), index of s_i-1 in dep_mat < s_i-k
        for map in self.maps_lst:
            map_h = map.shape[0]
            map_w = map.shape[1]

            self.count_each_tile(map, map_h, map_w)

            for D_idx in range(len(self.dep_matrices)):
                # Length of max row idx that contains 1 - min row idx that contains 1
                tile1_row_idx = np.where(np.any(self.dep_matrices[D_idx]==1, axis=1))[0]
                tile1_col_idx = np.where(np.any(self.dep_matrices[D_idx]==1, axis=0))[0]
                # Only one tile 2 in D
                tile2_row_idx = min(np.where(np.any(self.dep_matrices[D_idx]==2, axis=1))[0])
                tile2_col_idx = min(np.where(np.any(self.dep_matrices[D_idx]==2, axis=0))[0])

                dep_mat_row_orig_idx = min(min(tile1_row_idx), tile2_row_idx)
                dep_mat_col_orig_idx = min(min(tile1_col_idx), tile2_col_idx)

                if self.learning_dr == 'top-down':
                    if (max(tile1_row_idx) - min(tile1_row_idx) + 1) >= (tile2_row_idx - min(tile1_row_idx) + 1):
                        dep_h = max(tile1_row_idx) - min(tile1_row_idx) + 1
                    else:
                        dep_h = tile2_row_idx - min(tile1_row_idx) + 1

                    if (max(tile1_col_idx) - min(tile1_col_idx) + 1) >= (tile2_col_idx - min(tile1_col_idx) + 1):
                        dep_w = max(tile1_col_idx) - min(tile1_col_idx) + 1
                    else:
                        dep_w = tile2_col_idx - min(tile1_col_idx) + 1

                    minimum_D_mat = self.dep_matrices[D_idx][dep_mat_row_orig_idx:dep_mat_row_orig_idx+dep_h, dep_mat_col_orig_idx:dep_mat_col_orig_idx+dep_w]

                    for row_i in range(0, map_h-dep_h+1):
                        for col_j in range(0, map_w-dep_w+1):
                            self.count_submap_config(map, minimum_D_mat, D_idx, row_i, col_j, dep_h, dep_w)

                if self.learning_dr == 'bottom-up':
                    if (max(tile1_row_idx) - min(tile1_row_idx) + 1) >= (max(tile1_row_idx) - tile2_row_idx + 1):
                        dep_h = max(tile1_row_idx) - min(tile1_row_idx) + 1
                    else:
                        dep_h = max(tile1_row_idx) - tile2_row_idx + 1

                    if (max(tile1_col_idx) - min(tile1_col_idx) + 1) >= (tile2_col_idx - min(tile1_col_idx) + 1):
                        dep_w = max(tile1_col_idx) - min(tile1_col_idx) + 1
                    else:
                        dep_w = tile2_col_idx - min(tile1_col_idx) + 1

                    minimum_D_mat = self.dep_matrices[D_idx][dep_mat_row_orig_idx:dep_mat_row_orig_idx+dep_h, dep_mat_col_orig_idx:dep_mat_col_orig_idx+dep_w]

                    for row_i in range(map_h-dep_h, -1, -1):
                        for col_j in range(0, map_w-dep_w+1):
                            self.count_submap_config(map, minimum_D_mat, D_idx, row_i, col_j, dep_h, dep_w)

    def count_each_tile(self, map, map_h, map_w):
        for r_i in range(map_h):
            for c_j in range(map_w):
                if map[r_i][c_j] not in self.each_tile_abs_counts:
                    self.each_tile_abs_counts[map[r_i][c_j]] = 1
                else:
                    self.each_tile_abs_counts[map[r_i][c_j]] += 1

    def count_submap_config(self, map, dep_mat, D_idx, row_i, col_j, dep_h, dep_w):
        sub_map = map[row_i:row_i+dep_h, col_j:col_j+dep_w]

        tile_2_mark = dep_mat == 2
        tile_1_mark = dep_mat == 1

        tile_2_val = None
        tile_1s_val = []

        for sub_r_i in range(dep_h):
            for sub_c_j in range(dep_w):
                if dep_mat[sub_r_i][sub_c_j] == 1:
                    tile_1s_val.append(sub_map[sub_r_i][sub_c_j])

                if dep_mat[sub_r_i][sub_c_j] == 2:
                    tile_2_val = sub_map[sub_r_i][sub_c_j]

        type_comb = tile_2_val
        for tile1 in tile_1s_val:
            type_comb += tile1

        if type_comb not in self.abs_counts:
            self.abs_counts[D_idx][type_comb] = 1
        else:
            self.abs_counts[D_idx][type_comb] += 1

    def prob_Estimation(self):
        total_tile_counts = sum(self.each_tile_abs_counts.values())

        for tile in self.each_tile_abs_counts:
            self.each_tile_prob_distr[tile] = self.each_tile_abs_counts[tile] / total_tile_counts

        for D_key in self.abs_counts:
            # Given the number of occurences of tile2 and tile1s, get CPD P(tile2 | tile1s) = T(tile2, tile1s) / sum_all_tile2s(T(tile2, tile1s))
            tile1s_config = dict()
            for tiles_config in self.abs_counts[D_key]:
                # Count the number of times tile1s config occurs
                if tiles_config[1:] not in tile1s_config:
                    tile1s_config[tiles_config[1:]] = 1
                else:
                    tile1s_config[tiles_config[1:]] += 1

            for tiles_config in self.abs_counts[D_key]:
                self.cond_prob_distr[D_key][tiles_config] = self.abs_counts[D_key][tiles_config] / tile1s_config[tiles_config[1:]]

    def sampling(self, g_map_w, g_map_h):
        pass



tileTypes = {
    "F": "FLOOR",
    "B": "BLOCK",
    "M": "MONSTER",
    "P": "ELEMENT (LAVA, WATER)",
    "O": "ELEMENT + FLOOR (LAVA/BLOCK, WATER/BLOCK)",
    "I": "ELEMENT + BLOCK",
    "D": "DOOR",
    "S": "STAIR",
    "W": "WALL",
    "-": "VOID"
}

maps_lst = readMaps("./maps", tileTypes)

# From left to right to generate each row
D_1 = np.asarray([[0, 0, 0], [0, 0, 0], [0, 1, 2]], dtype=np.int8)
D_2 = np.asarray([[0, 0, 0], [0, 0, 1], [0, 1, 2]], dtype=np.int8)
D_3 = np.asarray([[0, 0, 0], [0, 0, 0], [1, 1, 2]], dtype=np.int8)
D_4 = np.asarray([[0, 0, 0], [0, 1, 0], [0, 1, 2]], dtype=np.int8)
D_5 = np.asarray([[0, 0, 0], [0, 1, 1], [0, 1, 2]], dtype=np.int8)
D_6 = np.asarray([[0, 0, 0], [0, 0, 1], [1, 1, 2]], dtype=np.int8)
D = [D_1, D_2, D_3, D_4, D_5, D_6]
learning_dr = 'top-down'

# D_1 = np.asarray([[0, 0, 0], [0, 0, 0], [0, 1, 2]], dtype=np.int8)
# D_2 = np.asarray([[0, 0, 0], [0, 1, 2], [0, 0, 1]], dtype=np.int8)
# D_3 = np.asarray([[0, 0, 0], [0, 0, 0], [1, 1, 2]], dtype=np.int8)
# D_4 = np.asarray([[0, 0, 0], [0, 1, 2], [0, 1, 0]], dtype=np.int8)
# D_5 = np.asarray([[0, 0, 0], [0, 1, 2], [0, 1, 1]], dtype=np.int8)
# D_6 = np.asarray([[0, 0, 0], [1, 1, 2], [0, 0, 1]], dtype=np.int8)
# D = [D_1, D_2, D_3, D_4, D_5, D_6]
# learning_dr = 'bottom-up'

# D_5
dep_mat_idx = 5

basic_MC = basicMarkovChain(maps_lst, D, learning_dr, dep_mat_idx)
# Learning
basic_MC.absolute_Counts()
basic_MC.prob_Estimation()
# print(basic_MC.abs_counts)
# print(basic_MC.each_tile_abs_counts)
# print(basic_MC.cond_prob_distr)
# print(basic_MC.each_tile_prob_distr)
# Generation
