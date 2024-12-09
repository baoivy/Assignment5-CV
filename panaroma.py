import numpy as np
import cv2

def image_stitch(images, lowe_ratio=0.75, max_Threshold=4.0, feature_detector="orb"):
    # Chọn bộ phát hiện đặc trưng
    if feature_detector == "sift":
        descriptors = cv2.SIFT_create()
    else:
        descriptors = cv2.ORB_create()

    (imageB, imageA) = images
    (key_points_A, features_of_A) = detect_feature_and_keypoints(descriptors, imageA)
    (key_points_B, features_of_B) = detect_feature_and_keypoints(descriptors, imageB)

    Homography = find_homography(key_points_A, key_points_B, features_of_A, features_of_B, lowe_ratio, max_Threshold)
    result_image = cv2.warpPerspective(imageA, Homography, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    result_image[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

    return result_image

def detect_feature_and_keypoints(descriptors, image):
    (keypoints, features) = descriptors.detectAndCompute(image, None)
    keypoints = np.float32([i.pt for i in keypoints])
    return keypoints, features

def get_all_valid_matches(AllMatches, lowe_ratio):
    valid_matches = []
    for val in AllMatches:
        if len(val) == 2 and val[0].distance < val[1].distance * lowe_ratio:
            valid_matches.append((val[0].trainIdx, val[0].queryIdx))
    return valid_matches

def find_homography(KeypointsA, KeypointsB, featuresA, featuresB, lowe_ratio, max_Threshold):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING if featuresA.dtype == np.uint8 else cv2.NORM_L2, crossCheck=False)
    all_matches = bf.knnMatch(featuresA, featuresB, k=2)
    valid_matches = get_all_valid_matches(all_matches, lowe_ratio)

    if len(valid_matches) <= 4:
        return None

    points_A = np.float32([KeypointsA[i] for (_, i) in valid_matches])
    points_B = np.float32([KeypointsB[i] for (i, _) in valid_matches])

    # Tính toán ma trận Homography
    Homography, _ = cv2.findHomography(points_A, points_B, cv2.RANSAC, max_Threshold)
    return Homography

def get_points(imageA, imageB):
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA], vis[0:hB, wA:] = imageA, imageB
    return vis
