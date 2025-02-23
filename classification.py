import os
import cv2
import json
import numpy as np
from skimage.feature import hog

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
    return {"Keypoints": len(keypoints), "Descriptors": descriptors.tolist() if descriptors is not None else []}

def region_descriptor(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    descriptors = []
    for contour in contours:
        area = cv2.contourArea(contour)
        moments = cv2.moments(contour)
        centroid = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])) if moments['m00'] != 0 else (0, 0)
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

# --- Retrieval function ---
def retrieve_images(query_vector, per_image_scores):
    results = []
    sift_scores = per_image_scores.get("SIFT", {})
    for category, images in sift_scores.items():
        for filename, stored_sift in images.items():
            stored_region   = per_image_scores.get("Region", {}).get(category, {}).get(filename, 0.0)
            stored_boundary = per_image_scores.get("Boundary", {}).get(category, {}).get(filename, 0.0)
            stored_hog      = per_image_scores.get("HOG", {}).get(category, {}).get(filename, 0.0)
            stored_vector = np.array([stored_sift, stored_region, stored_boundary, stored_hog])
            distance = np.linalg.norm(query_vector - stored_vector)
            results.append((category, filename, distance, stored_vector))
    results.sort(key=lambda item: item[2])
    return results

def display_results(results_list):
    print("\n🏆 Top 5 Overall Matches:")
    for idx, (category, filename, distance, _) in enumerate(results_list[:5], start=1):
        print(f"{idx}. Category: {category}, File: {filename}, Distance: {distance:.3f}")
    print("\n📂 Best Match per Category:")
    best_per_category = {}
    for category, filename, distance, _ in results_list:
        if category not in best_per_category or distance < best_per_category[category][1]:
            best_per_category[category] = (filename, distance)
    for category, (filename, distance) in best_per_category.items():
        print(f"Category: {category}, Best Match: {filename}, Distance: {distance:.3f}")
    return best_per_category

def update_final_results(results_list):
    # Prepare a data structure for final results.
    final_top5 = []
    for idx, (category, filename, distance, _) in enumerate(results_list[:5], start=1):
        final_top5.append({
            "Rank": idx,
            "Category": category,
            "File": filename,
            "Distance": distance
        })
    best_per_category = {}
    for category, filename, distance, _ in results_list:
        if category not in best_per_category or distance < best_per_category[category]["Distance"]:
            best_per_category[category] = {"File": filename, "Distance": distance}
    final_results = {
        "Top5OverallMatches": final_top5,
        "BestMatchPerCategory": best_per_category
    }
    return final_results

def save_final_results(final_results, filename="final_results.json"):
    with open(filename, "w") as outfile:
        json.dump(final_results, outfile, indent=4)
    print(f"\nFinal retrieval results updated and saved to '{filename}'.")

def main():
    # --- Step 1. Load and process the query image ---
    query_filename = "query.jpg"
    query_image_path = os.path.join(os.getcwd(), query_filename)
    if not os.path.exists(query_image_path):
        print(f"❌ Query image '{query_filename}' not found in the current folder.")
        return

    query_image = cv2.imread(query_image_path)
    if query_image is None:
        print("❌ Error reading the query image.")
        return

    query_image = preprocess_image(query_image)
    query_features = extract_features(query_image)

    query_sift = compute_sift_score(query_features["SIFT"])
    query_region = compute_region_score(query_features["Region"])
    query_boundary = compute_boundary_score(query_features["Boundary"])
    query_hog = compute_hog_score(query_features["HOG"])

    query_vector = np.array([query_sift, query_region, query_boundary, query_hog])
    print("🔍 Initial Query Image Scores:")
    print(f"   SIFT: {query_sift}")
    print(f"   Region: {query_region}")
    print(f"   Boundary: {query_boundary}")
    print(f"   HOG: {query_hog}\n")

    # --- Step 2. Load the precomputed scores from the JSON file ---
    json_filename = "feature_scores.json"
    json_file_path = os.path.join(os.getcwd(), json_filename)
    if not os.path.exists(json_file_path):
        print(f"❌ Feature scores file '{json_filename}' not found in the current folder.")
        return

    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)
    per_image_scores = data.get("PerImageScores", {})

    # --- Step 3. Initial retrieval ---
    results = retrieve_images(query_vector, per_image_scores)
    display_results(results)

    # Update final results file after the initial retrieval.
    final_results = update_final_results(results)
    save_final_results(final_results)

    # --- Step 4. Relevance feedback loop ---
    alpha = 0.7
    beta = 0.3

    while True:
        feedback = input("\nWould you like to provide relevance feedback? (y/n): ").strip().lower()
        if feedback != "y":
            print("Terminating relevance feedback.")
            break

        indices_str = input("Enter the indices (comma separated) of the relevant images from the top 5: ").strip()
        try:
            indices = [int(i.strip()) for i in indices_str.split(",") if i.strip().isdigit()]
        except Exception as e:
            print("Invalid input. Please enter valid numeric indices separated by commas.")
            continue

        valid_indices = [i for i in indices if 1 <= i <= 5]
        if not valid_indices:
            print("No valid indices entered. Please enter numbers between 1 and 5.")
            continue

        relevant_vectors = []
        for i in valid_indices:
            try:
                relevant_vectors.append(results[i-1][3])
            except IndexError:
                pass

        if not relevant_vectors:
            print("No relevant images found for the provided indices.")
            continue

        # Compute the average vector of the relevant images.
        avg_relevant_vector = np.mean(relevant_vectors, axis=0)
        new_query_vector = alpha * query_vector + beta * avg_relevant_vector
        print("\nUpdated query vector (after feedback):", new_query_vector)

        # Update the query vector for subsequent retrievals.
        query_vector = new_query_vector
        results = retrieve_images(query_vector, per_image_scores)
        display_results(results)

        # Update the final results file after each relevance feedback iteration.
        final_results = update_final_results(results)
        save_final_results(final_results)

    print("\nFinal retrieval results:")
    best_per_category = display_results(results)
    final_results = update_final_results(results)
    save_final_results(final_results)

if __name__ == "__main__":
    main()