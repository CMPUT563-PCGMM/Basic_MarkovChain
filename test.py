import mapProcessor as mp 
import eval
import img_gen as ig

if __name__ == "__main__":
    maps_data = mp.readMaps("../data/TheLegendofZelda/Processed")
    rooms = mp.roomSplit(maps_data)
    training_data, validation_data, testing_data = mp.data_split(rooms)
    imgs_dic = ig.getAllTileImg("../data/pics")

    for room in rooms:
        ig.showRoom(room,imgs_dic)
        # a = input()