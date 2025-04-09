import cv2
import os
from datetime import datetime
import numpy as np
import easyocr
from ultralytics import YOLO  # Correct way to import YOLOv8

def detect_and_extract_text_from_folder(image_folder, model_path, save_dir):
    # Load YOLOv8 model (using GPU)
    model = YOLO(model_path)

    # Initialize EasyOCR reader (using GPU)
    reader = easyocr.Reader(['en'], gpu=True)

    # Create directory to save number plates if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    extracted_texts = {}

    # Loop through all images in the folder
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)

        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue  # Skip non-image files

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read {image_path}. Skipping...")
            continue

        # Perform detection
        results = model(image)[0]

        # Process detections
        extracted_texts[image_name] = []
        for i, result in enumerate(results.boxes.data.tolist()):
            x1, y1, x2, y2, score, class_id = result

            if score > 0.5:  # Detection threshold
                # Convert coordinates to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Crop the detected number plate
                number_plate = image[y1:y2, x1:x2]

                # Convert to grayscale
                gray_plate = cv2.cvtColor(number_plate, cv2.COLOR_BGR2GRAY)

                # Extract text using EasyOCR
                result = reader.readtext(gray_plate)
                extracted_text = ' '.join([res[1] for res in result])
                extracted_texts[image_name].append(extracted_text.strip())

                # Generate a unique filename with date and time
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = os.path.join(save_dir, f'{image_name}_plate_{timestamp}_{i}.png')

                # Save the cropped number plate
                cv2.imwrite(filename, number_plate)
                print(f'Saved number plate: {filename} | Extracted Text: {extracted_text.strip()}')

    return extracted_texts


if __name__ == "__main__":
    # Define paths
    model_path = r"C:\Users\amarn\myenv\Scripts\runs\detect\train6\weights\last.pt"
    image_folder = r"P:\jindal project\images" # Folder containing images
    save_dir = r"P:\jindal project\cropped img1"  # Directory to save cropped number plates

    # Run detection and text extraction
    extracted_texts = detect_and_extract_text_from_folder(image_folder, model_path, save_dir)

    # Print all extracted texts
    for image_name, texts in extracted_texts.items():
        for text in texts:
            print(f'{image_name} - Extracted Text: {text}')
