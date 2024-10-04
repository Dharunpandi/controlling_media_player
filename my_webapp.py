import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

model = load_model('handrecognition_model.hdf5')

if 'stop' not in st.session_state:
    st.session_state.stop = False

def run_gesture_recognition():
    st.title("Hand Gesture Recognition")
    video_placeholder = st.empty()
    cap = cv2.VideoCapture(0)

    # Loop for video capture
    while not st.session_state.stop:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video.")
            break
        
        roi = frame[100:300, 100:300]
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        processed_roi = cv2.resize(gray_roi, (120, 120))
        
        processed_roi = processed_roi.reshape(1, 120, 120, 1).astype('float32') 
        
        processed_roi /= 255.0

        prediction = model.predict(processed_roi)
        gesture = np.argmax(prediction)
        confidence = np.max(prediction)
        
        cv2.putText(frame, f"Gesture: {gesture}, Confidence: {confidence:.2f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels='RGB')

    cap.release()
    cv2.destroyAllWindows()

    st.session_state.stop = False

if __name__ == "__main__":
    if st.button("Stop"):
        st.session_state.stop = True
    
    run_gesture_recognition()
