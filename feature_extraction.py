import cv2


def preprocess_image(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    return image


def sift_feature_extraction(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    # Save only the number of keypoints (score) and, if needed, a subset of descriptors.
    return {"Keypoints": len(keypoints), "Descriptors": descriptors.tolist() if descriptors is not None else []}

def region_descriptor(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    descriptors = []
    for contour in contours:
        area = cv2.contourArea(contour)
        moments = cv2.moments(contour)
        centroid = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])) if moments[
                                                                                                       'm00'] != 0 else (
        0, 0)
        descriptors.append({'Area': area, 'Centroid': centroid})
    return descriptors

def boundary_descriptor(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    descriptors = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        compactness = (perimeter ** 2) / (4 * np.pi * area) if area != 0 else 0
        descriptors.append({'Perimeter': perimeter, 'Compactness': compactness})
    return descriptors

def hog_feature_extraction(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    hog_features, _ = hog(image,
                          orientations=orientations,
                          pixels_per_cell=pixels_per_cell,
                          cells_per_block=cells_per_block,
                          block_norm='L2-Hys',
                          visualize=True,
                          transform_sqrt=True)
    return hog_features.tolist()
