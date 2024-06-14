import numpy as np
from ultralytics import YOLO
import cv2
import threading
import streamlit as st
from PIL import Image
import tempfile
import io

def classify_image(image_path, results_dict):
    # Process 1: Classification
    model_classification = YOLO('classification/runs/classify/train/weights/best.pt')
    predict = model_classification(image_path)
    names_dict = predict[0].names
    output = predict[0].probs.data.tolist()
    classification_result = "The image is: " + names_dict[np.argmax(output)]
    results_dict['classification'] = classification_result
    results_dict['classification_label'] = names_dict[np.argmax(output)]

def detect_objects(image_path, results_dict):
    # Process 2: Detection with bounding box width display
    model_detection = YOLO('detection/runs/detect/train/weights/best.pt')
    results = model_detection(image_path)

    # Load image
    image = cv2.imread(image_path)

    text_location = 'y1'  # default
    if 'classification_label' in results_dict and results_dict['classification_label'] == 'Mandible':
        text_location = 'y2'

    # Loop through detections and draw bounding boxes with widths
    for detection in results[0].boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Bounding box coordinates
        width = x2 - x1  # Calculate width
        label = f'{width}px'  # Label with width in pixels

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 100, 225), 12)
        
        text_y = y1 if text_location == 'y1' else y2
        text_y -= 10  # Offset for the text position
        # Display width on the image
        cv2.rectangle(image, (x1+30, text_y-50), (x1+220, text_y+20), (0, 0, 0), -1)
        cv2.putText(image, label, (x1+40, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 5)

    # Encode image to PNG
    _, encoded_image = cv2.imencode('.png', image)
    results_dict['detection_image'] = encoded_image

def main():
    st.title('Teeth Classification and Length Estimator')

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_image_path = temp_file.name
        
        # Dictionary to store results
        results_dict = {}

        # Create threads for classification and detection
        thread1 = threading.Thread(target=classify_image, args=(temp_image_path, results_dict))
        thread2 = threading.Thread(target=detect_objects, args=(temp_image_path, results_dict))

        # Start threads
        thread1.start()
        thread2.start()

        # Wait for both threads to finish
        thread1.join()
        thread2.join()

        # Display results with Streamlit
        if 'classification' in results_dict:
            st.subheader('Classification Result')
            st.write(results_dict['classification'])

        if 'detection_image' in results_dict:
            st.subheader('Detection Result')
            detection_image = Image.open(io.BytesIO(results_dict['detection_image']))
            st.image(detection_image, caption='Detected Objects with Bounding Box Widths')

if __name__ == '__main__':
    main()
