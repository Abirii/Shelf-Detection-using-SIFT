import cv2
import numpy as np


sift = cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 50)
search_params = dict(checks=100)


def shelf_detection(items_keypoint, patches, patche_info, shelfs_color):
    print('FLANN Matcher on each patche')

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    for k in range(len(items_keypoint)):
        obj_img, obj_keypoint, obj_descriptors = items_keypoint[k]  # get all items

        for i in range(len(patches)):

                patche_keypoints, patche_descriptors = sift.detectAndCompute(patches[i], None)
                try:
                    matches = flann.knnMatch(obj_descriptors, patche_descriptors, k=2)
                except:
                    continue

                matches_num = 0
                for j, (m, n) in enumerate(matches):
                    if m.distance < 0.55 * n.distance:
                        matches_num = matches_num + 1
                        #print(matches_num)

                if matches_num >= 3:
                    x, y, w, h = patche_info[i]
                    shelfs_color = cv2.resize(shelfs_color, (900, 900))
                    cv2.rectangle(shelfs_color, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)

    return shelfs_color