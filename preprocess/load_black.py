# function
# output = contour2mono.contour2mono(input)
# input is a numpy array in bgr color mode
# with red contour and black area
# output is a numpy array in bgr color mode
# with white nuclei and black background

import numpy as np
import cv2
from skimage.morphology import label

class BlackPixelFiner:

    def __init__(self):
        pass

    def __fine_nearby_black_pixel(self, x, y):
        i, j = np.where(self.map[x-1:x+2, y-1:y+2] == 1)
        for t in range(len(i)):
            self.__fine_black_pixel(x-1+i[t], y-1+j[t])

    def __fine_black_pixel(self, x, y):
        edge_or_corner = [x == 0, x == self.map.shape[0],
                          y == 0, y == self.map.shape[1]]
        if np.any(edge_or_corner):
            return
        # 0 for nuclei, 1 for backgroun

        # 000
        # 010
        if np.sum(self.map[x-1:x+1, y-1:y+2]) == 1:
            self.map[x][y] = 0
            self.__fine_nearby_black_pixel(x,y)
            return
        # 010
        # 000
        if np.sum(self.map[x:x+2, y-1:y+2]) == 1:
            self.map[x][y] = 0
            self.__fine_nearby_black_pixel(x,y)
            return
        # 00
        # 01
        # 00
        if np.sum(self.map[x-1:x+2, y-1:y+1]) == 1:
            self.map[x][y] = 0
            self.__fine_nearby_black_pixel(x,y)
            return
        # 00
        # 10
        # 00
        if np.sum(self.map[x-1:x+2, y:y+2]) == 1:
            self.map[x][y] = 0
            self.__fine_nearby_black_pixel(x,y)
            return
        return

    def apply(self, map):
        self.map = map
        # only deal with the pixel which is not at the edge or corner
        for i in range(1,self.map.shape[0]-1):
            for j in range(1,self.map.shape[1]-1):
                self.__fine_black_pixel(i, j)

class ContourEdgeFixer:

    def __init__(self):
        pass

    def __check_red_pixel(self):
        # 0 for area, -1 for contour

        # 000
        # 010
        if np.sum(self.map[self.red_x-1:self.red_x+1,
                           self.red_y-1:self.red_y+2]) == -1:
            self.map[self.edge_x][self.edge_y] = -1
            return
        # 010
        # 000
        if np.sum(self.map[self.red_x:self.red_x+2,
                           self.red_y-1:self.red_y+2]) == -1:
            self.map[self.edge_x][self.edge_y] = -1
            return
        # 00
        # 01
        # 00
        if np.sum(self.map[self.red_x-1:self.red_x+2,
                           self.red_y-1:self.red_y+1]) == -1:
            self.map[self.edge_x][self.edge_y] = -1
            return
        # 00
        # 10
        # 00
        if np.sum(self.map[self.red_x-1:self.red_x+2,
                           self.red_y:self.red_y+2]) == -1:
            self.map[self.edge_x][self.edge_y] = -1
            return
        return

    def apply(self, map):

        self.map = map

        # check left edge
        # me -1
        self.edge_x = 0
        self.edge_y = 0
        self.red_x = 0
        self.red_y = 1
        for i in range(1,self.map.shape[0]-1):
            if self.map[i][self.red_y] == -1:
                self.edge_x = i
                self.red_x = i
                self.__check_red_pixel()
        # check right edge
        self.edge_x = 0
        self.edge_y = self.map.shape[1] - 1
        self.red_x = 0
        self.red_y = self.map.shape[1] - 2
        for i in range(1,self.map.shape[0]-1):
            if self.map[i][self.red_y] == -1:
                self.edge_x = i
                self.red_x = i
                self.__check_red_pixel()
        # check top edge
        self.edge_x = 0
        self.edge_y = 0
        self.red_x = 1
        self.red_y = 0
        for i in range(1,self.map.shape[1]-1):
            if self.map[self.red_x][i] == -1:
                self.edge_y = i
                self.red_y = i
                self.__check_red_pixel()
        # check bottom edge
        self.edge_x = self.map.shape[0] -1
        self.edge_y = 0
        self.red_x = self.map.shape[0] - 2
        self.red_y = 0
        for i in range(1,self.map.shape[1]-1):
            if self.map[self.red_x][i] == -1:
                self.edge_y = i
                self.red_y = i
                self.__check_red_pixel()
        return



def contour2mono(img):

    # change a bgr to only one digit
    # -1 for contour
    map = np.zeros(img.shape[0:2], np.int8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][2] == 255:
                map[i][j] = -1

    # optinal (maybe useless toward newer algorithm)
    # contour edge fix
    # disable it by simply delete the following 2 lines of code
    fixer = ContourEdgeFixer()
    fixer.apply(map)

    # labeling
    map, n_labels = label(map, background=-1, connectivity=1,return_num=True)


    # count area
    area_list = []
    for i in range(n_labels):
        area_list.append(np.sum(map == i))
    area_list = np.array(area_list)


    # distinguish area
    max_noise_size = 100
    big_indices = np.array(np.where(area_list > max_noise_size))
    # delete the background(biggest area) & contour
    nuclei_indices = np.delete(big_indices, (area_list.argmax(), 0))

    # convert to nuclei and background only
    # 0 for nuclei, 1 for background
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if np.any(nuclei_indices == map[i][j]):
                map[i][j] = 0
            else:
                map[i][j] = 1

    # optional (cost lots of time)
    # usage: fining
    # detect some black pixels which are obviously strange
    # disable it by simply delete the following 2 lines of code
    finer = BlackPixelFiner()
    finer.apply(map)



    # change 1digit pic(map) to bgr in only black & white color
    output = np.zeros(img.shape, np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if map[i][j] == 0:
                output[i][j] = [255,255,255]

    return output

# code for test

# img = cv2.imread('test.png')
# output = contour2mono(img)
# cv2.imshow('image',output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
