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

def extract_features(image):
    sift = sift_feature_extraction(image)
    region = region_descriptor(image)
    boundary = boundary_descriptor(image)
    hog_feats = hog_feature_extraction(image)
    return {
        "SIFT": sift,
        "Region": region,
        "Boundary": boundary,
        "HOG": hog_feats
    }

def compute_sift_score(sift_data):
    return sift_data.get("Keypoints", 0)


def compute_region_score(region_data):
    if not region_data:
        return 0.0
    areas = [r["Area"] for r in region_data]
    return float(np.mean(areas))


def compute_boundary_score(boundary_data):
    if not boundary_data:
        return 0.0
    compactness_values = [b["Compactness"] for b in boundary_data]
    return float(np.mean(compactness_values))


def compute_hog_score(hog_data):
    if not hog_data:
        return 0.0
    return float(np.mean(hog_data))

def run_feature_extraction(base_folder, output_json_path):
    scores = {
        "SIFT": {},
        "Region": {},
        "Boundary": {},
        "HOG": {}
    }

    all_scores = {
        "SIFT": [],
        "Region": [],
        "Boundary": [],
        "HOG": []
    }

    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"❌ Unable to read {file}. Skipping.")
                    continue

                image = preprocess_image(image)
                relative_path = os.path.relpath(image_path, base_folder)
                category = relative_path.split(os.sep)[0]

                features = extract_features(image)

                sift_score = compute_sift_score(features["SIFT"])
                region_score = compute_region_score(features["Region"])
                boundary_score = compute_boundary_score(features["Boundary"])
                hog_score = compute_hog_score(features["HOG"])

                if category not in scores["SIFT"]:
                    scores["SIFT"][category] = {}
                    scores["Region"][category] = {}
                    scores["Boundary"][category] = {}
                    scores["HOG"][category] = {}

                scores["SIFT"][category][file] = sift_score
                scores["Region"][category][file] = region_score
                scores["Boundary"][category][file] = boundary_score
                scores["HOG"][category][file] = hog_score

                all_scores["SIFT"].append(sift_score)
                all_scores["Region"].append(region_score)
                all_scores["Boundary"].append(boundary_score)
                all_scores["HOG"].append(hog_score)

                print(f"✅ Processed {relative_path}")

    statistics = {}
    for feature_type in ["SIFT", "Region", "Boundary", "HOG"]:
        scores_array = np.array(all_scores[feature_type])
        if scores_array.size > 0:
            avg = float(np.mean(scores_array))
            var = float(np.var(scores_array))
        else:
            avg = 0.0
            var = 0.0
        statistics[feature_type] = {"Average": avg, "Variance": var}

    output_data = {
        "PerImageScores": scores,
        "GlobalStatistics": statistics
    }

    with open(output_json_path, "w") as json_file:
        json.dump(output_data, json_file, indent=4)
        print("Scores saved successfully.")

    print(f"✅ Scores and statistics saved to {output_json_path}")


if __name__ == "__main__":
    base_folder = os.getcwd()
    output_json_path = os.path.join(base_folder, "feature_scores.json")
    run_feature_extraction(base_folder, output_json_path)