# modules/image_processing.py

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import tflite_runtime.interpreter as tflite

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def load_tflite_model(model_path):
    try:
        interpreter = tflite.Interpreter(model_path=model_path, num_threads=4)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_image(interpreter, image):
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        image_input = np.expand_dims(image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], image_input)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])
        prob = output_data[0][0]
        return "Drunk" if prob > 0.7 else "Not Drunk"
    except Exception as e:
        print(f"Error predicting image: {e}")
        return "Not Drunk"

def preprocess_image(image, target_size=(224, 224)):
    try:
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_img)
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0]
        face_oval = mp.solutions.face_mesh.FACEMESH_FACE_OVAL
        df = pd.DataFrame(list(face_oval), columns=["p1", "p2"])
        routes_idx = [(df.iloc[i]["p1"], df.iloc[i]["p2"]) for i in range(len(df))]
        routes = []
        for source_idx, target_idx in routes_idx:
            source = landmarks.landmark[source_idx]
            target = landmarks.landmark[target_idx]
            routes.append((int(image.shape[1] * source.x), int(image.shape[0] * source.y)))
            routes.append((int(image.shape[1] * target.x), int(image.shape[0] * target.y)))
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(routes, dtype=np.int32)], 255)
        face_extracted = cv2.bitwise_and(image, image, mask=mask)
        x_min, y_min = np.min(routes, axis=0)
        x_max, y_max = np.max(routes, axis=0)
        cropped = face_extracted[y_min:y_max, x_min:x_max]
        if cropped.size == 0:
            return None
        image_resized = cv2.resize(cropped, target_size)
        image_normalized = image_resized / 255.0
        return image_normalized
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None
