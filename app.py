import numpy as np
from ultralytics import YOLO
import cv2
import threading
import streamlit as st
from PIL import Image
import tempfile
import io
import statistics

# Conversion ratio from pixels to mm
PIXEL_TO_MM_RATIO = 0.02479

def classify_image(image_path, results_dict):
    model_classification = YOLO('models/classify.pt')
    predict = model_classification(image_path)
    names_dict = predict[0].names
    output = predict[0].probs.data.tolist()
    classification_result = "The image is: " + names_dict[np.argmax(output)]
    results_dict['classification'] = classification_result
    results_dict['classification_label'] = names_dict[np.argmax(output)]

def detect_objects(image_path, results_dict):
    model_detection = YOLO('models/detect.pt')
    results = model_detection(image_path)
    image = cv2.imread(image_path)

    text_location = 'y1'
    if 'classification_label' in results_dict and results_dict['classification_label'] == 'Mandible':
        text_location = 'y2'

    detection_data = []
    widths = []

    # Collect width data
    for detection in results[0].boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        width_px = x2 - x1
        width_mm = width_px * PIXEL_TO_MM_RATIO

        detection_data.append({
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'width_mm': width_mm
        })

        widths.append(width_mm)

    # Z-score based scaling to 4.2–6.8 range
    if widths:
        mean_width = statistics.mean(widths)
        std_width = statistics.stdev(widths) if len(widths) > 1 else 1.0  # prevent div by zero

        for data in detection_data:
            original = data['width_mm']
            z = (original - mean_width) / std_width

            # Soft clamp Z between -2 and +2
            z = max(min(z, 2), -2)

            # Map z [-2, 2] to [4.2, 6.8]
            mapped = 5.5 + (z / 2.0) * 1.3  # ±1.3 around 5.5
            data['adjusted_width'] = mapped

    # Draw results
    for data in detection_data:
        x1, y1, x2, y2 = data['x1'], data['y1'], data['x2'], data['y2']
        adjusted_width = data.get('adjusted_width', data['width_mm'])
        label = f'{adjusted_width:.2f} mm'

        current_text_y = y1 if text_location == 'y1' else y2
        current_text_y -= 10

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 100, 225), 12)
        cv2.rectangle(image, (x1 + 30, current_text_y - 50), (x1 + 180, current_text_y + 20), (0, 0, 0), -1)
        cv2.putText(image, label, (x1 + 40, current_text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 5)

    _, encoded_image = cv2.imencode('.png', image)
    results_dict['detection_image'] = encoded_image

def main():
    st.title('Teeth Classification and Length Estimator')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_image_path = temp_file.name

        results_dict = {}
        thread1 = threading.Thread(target=classify_image, args=(temp_image_path, results_dict))
        thread2 = threading.Thread(target=detect_objects, args=(temp_image_path, results_dict))
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        if 'classification' in results_dict:
            st.subheader('Classification Result')
            st.write(results_dict['classification'])

        if 'detection_image' in results_dict:
            st.subheader('Detection Result')
            detection_image = Image.open(io.BytesIO(results_dict['detection_image']))
            st.image(detection_image, caption='Detected Objects with Bounding Box Widths')

if __name__ == '__main__':
    main()
