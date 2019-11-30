import cv2
import sys

'''GET DATA FUNCTIONS '''
def get_shelfs():
    shelfs = []
    for i in range(12):
        path = 'shelfs/' + str(i+1) + '.jpg'
        shelfs.append(cv2.imread(path))
    return shelfs

def get_items():
    items = []
    for i in range(6):
        path = 'items/' + str(i+1) + '.jpg'
        items.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    return items


''' GET DATA SIFTS FUNCTIONS'''
def get_items_SIFT(items):
    items_keypoint = []
    sift = cv2.xfeatures2d.SIFT_create()
    for i in range(len(items)):
        keypoints, descriptors = sift.detectAndCompute(items[i], None)
        item = cv2.drawKeypoints(items[i], keypoints, None)

        item_info = (item, keypoints, descriptors)
        items_keypoint.append(item_info)

        print('Item #' + str(i + 1))
    return items_keypoint

def get_shelfs_SIFT(shelfs):
    shelfs_keypoint = []
    sift = cv2.xfeatures2d.SIFT_create()
    for i in range(len(shelfs)):
        keypoints, descriptors = sift.detectAndCompute(shelfs[i], None)
        shelfs = cv2.drawKeypoints(shelfs[i], keypoints, None)

        shelfs_info = (shelfs, keypoints, descriptors)
        shelfs_keypoint.append(shelfs_info)

        print('shelf #' + str(i + 1))
    return shelfs_keypoint


''' After Selective Search we will get 400 bounding box, and then apply FLANN Matcher for each box. Then define threshold number, and if the number of matches pass the threshold number i will sign that box as "successful box".

NOTE: I have to reduce the image resolution and limit the number of bounding box to 400 because using CPU.

patche_info - (ROI, bounding boxs and there coordinates(x, y, w, h))'''
def selective_search(im):
    print('Get bounding boxs.')
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)

    patches_list = []
    patche_info = []

    im = cv2.resize(im, (900, 900))
    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    ss.setBaseImage(im)
    ss.switchToSelectiveSearchFast()
    #ss.switchToSelectiveSearchQuality()

    rects = ss.process()
    print('Total Number of Region Proposals: {}'.format(len(rects)))

    for i, rect in enumerate(rects):
        x, y, w, h = rect
        # crop and save bounding boxs and there coordinates
        patche_info.append((x, y, w, h))
        crop = im[y:y + h, x:x + w]
        # save only the patches with "Normal" size
        if (crop.shape[0] < 70) and (crop.shape[1] < 120 and crop.shape[1] > 100):
            patche_info.append((x, y, w, h))
            patches_list.append(crop)

    return (patches_list, patche_info)

def sliding_window(im):
    shelf = cv2.resize(im, (900, 900))

    windows = []
    windows_info = []

    stepSize = 100
    (width, height) = (100, 100)  # window size
    for x in range(0, shelf.shape[1] - width + 1, stepSize):
        for y in range(0, shelf.shape[0] - height + 1, stepSize):
            window = shelf[x:x + width, y:y + height, :]
            windows.append(window)
            windows_info.append((x, y, width, height))
            cv2.rectangle(shelf, (x, y), (x + width, y + height), (0, 255, 0), 2)
    return (windows, windows_info)