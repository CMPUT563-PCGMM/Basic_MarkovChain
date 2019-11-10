from PIL import Image
import numpy as np
import os
import mapProcessor as mp 

# w, h = 512, 512
# data = np.zeros((h, w, 3), dtype=np.uint8)
# data[256, 256] = [255, 0, 0]
# img = Image.fromarray(data, 'RGB')
# img.save('my.png')
# img.show()

name_dict = {   'block.png':'B',
                'door.png':'D',
                'floor.png':'F',
                'monstor.png':'M',
                'void.png':'-',
                'wall.png':'W',
                'water.png':'P'
            }

def getAllTileImg(path):
    imgs_dic = dict()
    for fileName in os.listdir(path):
        map_arr = readOneTileImg(path,fileName)
        imgs_dic[name_dict[fileName]] = map_arr
    return imgs_dic

def readOneTileImg(path,fileName):
    pic = np.asarray(Image.open(path + "/" + fileName))
    # pic2 = np.asarray(Image.open("../data/pics/floor.png"))
    # _all = np.append(pic,pic2,axis=0)
    # print(np.asarray(_all).shape)
    # img = Image.fromarray(pic, 'RGB')
    # img.show()
    return pic

def showRoom(room,imgs_dic):
    room_img = list()
    for i in range(room.shape[0]):
        line_img = list()
        for j in range(room.shape[1]):
            tile = imgs_dic[room[i][j]]
            if type(line_img) == list:
                line_img = tile
            else:
                line_img = np.append(line_img,tile, axis=0)
        if type(room_img) == list:
            room_img = line_img
        else:
            room_img = np.append(room_img,line_img, axis=1)
    img = Image.fromarray(room_img, 'RGB')
    img.show()
    return