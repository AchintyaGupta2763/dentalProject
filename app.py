import numpy as np
from ultralytics import YOLO
import cv2
import threading
import streamlit as st
from PIL import Image
import tempfile
import io
import statistics
import pandas as pd

# Conversion ratio from pixels to mm
PIXEL_TO_MM_RATIO = 0.02479

# ---------------------------
# 1) Hardcode your tables here
# ---------------------------
maxilla_data = {
    'Sum_of_Incisors': [19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0],
    '95%':             [21.6, 21.8, 22.1, 22.4, 22.7, 22.9, 23.2, 23.5, 23.8, 24.0, 24.3, 24.6],
    '85%':             [21.0, 21.3, 21.5, 21.8, 22.1, 22.4, 22.6, 22.9, 23.2, 23.5, 23.7, 24.0],
    '75%':             [20.6, 20.9, 21.2, 21.5, 21.8, 22.0, 22.3, 22.6, 22.9, 23.1, 23.4, 23.7],
    '65%':             [20.4, 20.6, 20.9, 21.2, 21.5, 21.8, 22.0, 22.3, 22.6, 22.8, 23.1, 23.4],
    '50%':             [20.0, 20.3, 20.6, 20.8, 21.1, 21.4, 21.7, 21.9, 22.2, 22.5, 22.8, 23.0],
    '35%':             [19.6, 19.9, 20.2, 20.5, 20.8, 21.0, 21.3, 21.6, 21.9, 22.1, 22.4, 22.7],
    '25%':             [19.4, 19.7, 19.9, 20.2, 20.5, 20.8, 21.0, 21.3, 21.6, 21.9, 22.1, 22.4],
    '15%':             [19.0, 19.3, 19.6, 19.9, 20.2, 20.4, 20.7, 21.0, 21.3, 21.5, 21.8, 22.1],
    '5%':              [18.5, 18.8, 19.0, 19.3, 19.6, 19.9, 20.1, 20.4, 20.7, 21.0, 21.2, 21.5]
}

mandible_data = {
    'Sum_of_Incisors': [19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0],
    '95%':             [21.1, 21.4, 21.7, 22.0, 22.3, 22.6, 22.9, 23.2, 23.5, 23.8, 24.1, 24.4],
    '85%':             [20.5, 20.8, 21.1, 21.4, 21.7, 22.0, 22.3, 22.6, 22.9, 23.2, 23.5, 23.8],
    '75%':             [20.1, 20.4, 20.7, 21.0, 21.3, 21.6, 21.9, 22.2, 22.5, 22.8, 23.1, 23.4],
    '65%':             [19.8, 20.1, 20.4, 20.7, 21.0, 21.3, 21.6, 21.9, 22.2, 22.5, 22.8, 23.1],
    '50%':             [19.4, 19.7, 20.0, 20.3, 20.6, 20.9, 21.2, 21.5, 21.8, 22.1, 22.4, 22.7],
    '35%':             [19.0, 19.3, 19.6, 19.9, 20.2, 20.5, 20.8, 21.1, 21.4, 21.7, 22.0, 22.3],
    '25%':             [18.7, 19.0, 19.3, 19.6, 19.9, 20.2, 20.5, 20.8, 21.1, 21.4, 21.7, 22.0],
    '15%':             [18.4, 18.7, 19.0, 19.3, 19.6, 19.8, 20.1, 20.4, 20.7, 21.0, 21.3, 21.6],
    '5%':              [17.7, 18.0, 18.3, 18.6, 18.9, 19.2, 19.5, 19.8, 20.1, 20.4, 20.7, 21.0]
}

df_maxilla = pd.DataFrame(maxilla_data)
df_mandible = pd.DataFrame(mandible_data)

# --------------------
# Original code below
# --------------------

@st.cache_resource
def load_models():
    return YOLO('models/classify.pt'), YOLO('models/detect.pt')

model_classification, model_detection = load_models()

def classify_image(image_path, results_dict):
    predict = model_classification(image_path)
    names_dict = predict[0].names
    output = predict[0].probs.data.tolist()
    classification_result = "The image is: " + names_dict[np.argmax(output)]
    results_dict['classification'] = classification_result
    results_dict['classification_label'] = names_dict[np.argmax(output)]

def detect_objects(image_path, results_dict):
    results = model_detection(image_path)
    image = cv2.imread(image_path)

    text_location = 'y1'
    if 'classification_label' in results_dict and results_dict['classification_label'] == 'Mandible':
        text_location = 'y2'

    detection_data = []
    widths = []

    for detection in results[0].boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        width_px = x2 - x1
        width_mm = width_px * PIXEL_TO_MM_RATIO

        detection_data.append({
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'width_mm': width_mm
        })
        widths.append(width_mm)

    if widths:
        mean_width = statistics.mean(widths)
        std_width = statistics.stdev(widths) if len(widths) > 1 else 1.0

        for data in detection_data:
            original = data['width_mm']
            z = (original - mean_width) / std_width
            z = max(min(z, 2), -2)
            mapped = 5.5 + (z / 2.0) * 1.3
            data['adjusted_width'] = mapped

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

def custom_round(val):
    whole = int(val)
    decimal = val - whole

    if decimal < 0.25:
        return float(whole)
    elif decimal < 0.75:
        return whole + 0.5
    else:
        return float(whole + 1)

def main():
    st.title('Teeth Classification and Length Estimator')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Initialize session state
    if 'results_dict' not in st.session_state:
        st.session_state.results_dict = {}
    if 'uploaded_filename' not in st.session_state:
        st.session_state.uploaded_filename = None

    results_dict = st.session_state.results_dict

    # Only run models if new file is uploaded
    if uploaded_file is not None:
        if st.session_state.uploaded_filename != uploaded_file.name:
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

            st.session_state.results_dict = results_dict
            st.session_state.uploaded_filename = uploaded_file.name
        else:
            results_dict = st.session_state.results_dict
    
    with st.container(border= True):
        st.markdown("### 🦷 Detection Section")
        if 'classification' in results_dict:
            st.info(f"**Classification Result:** {results_dict['classification']}")

        if 'detection_image' in results_dict:
            detection_image = Image.open(io.BytesIO(results_dict['detection_image']))
            st.image(detection_image, caption='Detected Objects with Bounding Box Widths')
        else:
            st.warning("Upload an image to see the detection result here.")
    
    st.markdown("---")

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            with st.container(height=420,border=True):
                st.markdown("### Manual Width Entry")
                # Empty text inputs to simulate blank number fields
                num1 = st.text_input("Enter Width 1 (mm)", value="", key="num1")
                num2 = st.text_input("Enter Width 2 (mm)", value="", key="num2")
                num3 = st.text_input("Enter Width 3 (mm)", value="", key="num3")
                num4 = st.text_input("Enter Width 4 (mm)", value="", key="num4")

        with col2:
            with st.container(height=420, border=True):
                st.markdown("### Find results")

                 # 2A) Add a dropdown to choose your percentage
                percentage_options = ["95%", "85%", "75%", "65%", "50%", "35%", "25%", "15%", "5%"]
                selected_percentage = st.selectbox("Select percentage:", percentage_options)

                if st.button("Calculate Total", use_container_width=True):
                    try:
                        values = [float(val) for val in [num1, num2, num3, num4] if val.strip() != ""]
                        total_sum = sum(values)
                        rounded_total = custom_round(total_sum)

                        colA, colB = st.columns(2)
                        with colA:
                            st.markdown(
                                f"<p style='font-size:17px; color:white;'><b>Total:</b> {total_sum:.2f} mm</p>", 
                                unsafe_allow_html=True
                            )

                        with colB:
                            st.markdown(
                                f"<p style='font-size:17px; color:white;'><b>Round off:</b> {rounded_total:.1f} mm</p>", 
                                unsafe_allow_html=True
                            )
                        
                        # 2B) Look up the row that’s closest to your Round off in Maxilla DF
                        # (If you want EXACT matching, you can do == instead of .abs().argsort() trick)
                        row_maxilla = df_maxilla.iloc[
                            (df_maxilla['Sum_of_Incisors'] - rounded_total).abs().argsort()[:1]
                        ]
                        # Get the chosen percentage value from that row
                        maxilla_value = row_maxilla[selected_percentage].values[0]

                        # 2C) Look up the row that’s closest to your Round off in Mandible DF
                        row_mandible = df_mandible.iloc[
                            (df_mandible['Sum_of_Incisors'] - rounded_total).abs().argsort()[:1]
                        ]
                        mandible_value = row_mandible[selected_percentage].values[0]

                        # 2D) Display the results from both tables
                        st.success(f"**Maxilla {selected_percentage} Value:** {maxilla_value:.1f}")
                        st.success(f"**Mandible {selected_percentage} Value:** {mandible_value:.1f}")

                    except ValueError:
                        st.error("Please enter valid numbers in all fields.")

if __name__ == '__main__':
    main()
