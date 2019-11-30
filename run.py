import cv2
import matcher
import numpy as np
import data_preprocessing


# get items and shelfs
print('Load data...')
items = data_preprocessing.get_items()
shelfs = data_preprocessing.get_shelfs()
if len(items) == 0 or len(shelfs) == 0:
    if len(shelfs) == 0:
        print('Cant load shelfs')
    else:
        print('Cant load items')
    exit(0)


# get items and shelfs SIFTS
print('Get items SIFTS:')
items_keypoint = data_preprocessing.get_items_SIFT(items)


# main loop
while True:
    print('Enter shelf number 1-12 or 0 to exit: ')

    shelf_num = int(input())
    shelf_num = shelf_num -1

    if shelf_num == -1:
        exit(0)

    shelf = shelfs[shelf_num]
    #patches, patche_info = data_preprocessing.sliding_window(shelf)
    patches, patche_info = data_preprocessing.selective_search(shelf)
    output = matcher.shelf_detection(items_keypoint, patches, patche_info, shelf)


    cv2.imshow("Matching result", output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
