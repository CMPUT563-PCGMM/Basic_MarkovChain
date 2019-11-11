import os

path = "Processed_changed_tiles"

output_path = "Processed_changed_tiles_reduced_OI"

reduced_tile_dict = {'O': 'F',
                     'I': 'B'}

for fileName in os.listdir(path):
    if fileName[:4] == "tloz":
        input_f = open(path+"/"+fileName, 'r')

        output_f = open(output_path+"/"+fileName, 'w')

        for row in input_f:
            row_chars = ''
            for char in row.rstrip():
                if char in reduced_tile_dict:
                    row_chars += reduced_tile_dict[char]
                else:
                    row_chars += char

            output_f.write(row_chars + '\n')
