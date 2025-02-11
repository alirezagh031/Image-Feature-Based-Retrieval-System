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