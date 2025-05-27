import cv2
import torch
import gradio as gr
import numpy as np
import tempfile

# Charger le modèle YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model.classes = [0]  # Classe 0 = "person"

# Fonctions de traitement d'image
def apply_processing(img, op):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if op == "Contours":
        return cv2.Canny(gray, 100, 200)
    elif op == "Histogramme":
        hist = cv2.calcHist([gray], [0], None, [256], [0,256])
        hist_img = np.zeros((300, 256, 3), dtype=np.uint8)
        cv2.normalize(hist, hist, 0, 300, cv2.NORM_MINMAX)
        for x, y in enumerate(hist):
            cv2.line(hist_img, (x, 300), (x, 300 - int(y)), (255,255,255), 1)
        return hist_img
    elif op == "Seuillage":
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return binary
    elif op == "Convolution":
        kernel = np.ones((5,5), np.float32)/25
        return cv2.filter2D(gray, -1, kernel)
    elif op == "Morphologie (dilatation)":
        kernel = np.ones((5,5), np.uint8)
        return cv2.dilate(gray, kernel, iterations=1)
    elif op == "Morphologie (érosion)":
        kernel = np.ones((5,5), np.uint8)
        return cv2.erode(gray, kernel, iterations=1)
    else:
        return gray

# Traitement principal de la vidéo
def process_video(video_path, traitement):
    cap = cv2.VideoCapture(video_path)
    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        boxes = results.xyxy[0]  # x1, y1, x2, y2, conf, class

        for *xyxy, conf, cls in boxes:
            x1, y1, x2, y2 = map(int, xyxy)
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            processed_roi = apply_processing(roi, traitement)

            # Si traitement a retourné une image mono (grayscale ou Canny), convert to BGR
            if len(processed_roi.shape) == 2:
                processed_roi = cv2.cvtColor(processed_roi, cv2.COLOR_GRAY2BGR)

            frame[y1:y2, x1:x2] = processed_roi

        if out is None:
            height, width = frame.shape[:2]
            out = cv2.VideoWriter(out_path, fourcc, 20.0, (width, height))

        out.write(frame)

    cap.release()
    out.release()
    return out_path

# Interface Gradio
interface = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Video(label="Vidéo d'entrée"),
        gr.Dropdown(
            choices=["Contours", "Histogramme", "Seuillage", "Convolution",
                     "Morphologie (dilatation)", "Morphologie (érosion)"],
            label="Traitement à appliquer sur les personnes détectées"
        )
    ],
    outputs=gr.Video(label="Vidéo traitée"),
    title="Vision par ordinateur sur des personnes détectées",
    description="Détection des personnes avec YOLOv5, puis traitement d'image appliqué uniquement à ces zones."
)

interface.launch(share=True)

