import os
from typing import Tuple, List, Dict

import cv2
import numpy as np


class FaceRecognizerService:
    def __init__(self, dataset_dir: str, trainer_dir: str) -> None:
        self.dataset_dir = dataset_dir
        self.trainer_dir = trainer_dir
        self.model_path = os.path.join(self.trainer_dir, "trainer.yml")
        self._recognizer = None
        self._face_detector = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        )

    def _ensure_recognizer_loaded(self) -> bool:
        if self._recognizer is None:
            self._recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
            if os.path.exists(self.model_path):
                self._recognizer.read(self.model_path)
        return self._recognizer is not None

    def save_face_image(self, student_id: int, raw_image_bytes: bytes) -> bool:
        image_array = np.frombuffer(raw_image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame is None:
            return False
        
        # Convert to grayscale and enhance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Better face detection parameters
        faces = self._face_detector.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=6, 
            minSize=(100, 100),
            maxSize=(300, 300)
        )
        
        if len(faces) == 0:
            return False

        # Save the largest detected face
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        face_roi = gray[y : y + h, x : x + w]
        
        # Ensure minimum size and resize to standard
        if w < 100 or h < 100:
            return False
            
        face_resized = cv2.resize(face_roi, (200, 200))

        student_dir = os.path.join(self.dataset_dir, f"student_{student_id}")
        os.makedirs(student_dir, exist_ok=True)
        next_index = self._next_image_index(student_dir)
        img_path = os.path.join(student_dir, f"img_{next_index:04d}.jpg")
        cv2.imwrite(img_path, face_resized)
        return True

    def _next_image_index(self, directory: str) -> int:
        indices = []
        for name in os.listdir(directory):
            if name.startswith("img_") and name.endswith(".jpg"):
                try:
                    idx = int(name[4:8])
                    indices.append(idx)
                except Exception:
                    pass
        return (max(indices) + 1) if indices else 1

    def train(self) -> Tuple[bool, int, int]:
        faces: List[np.ndarray] = []
        labels: List[int] = []
        for entry in os.listdir(self.dataset_dir):
            if not entry.startswith("student_"):
                continue
            try:
                student_id = int(entry.split("_")[1])
            except Exception:
                continue
            directory = os.path.join(self.dataset_dir, entry)
            for fname in os.listdir(directory):
                if not fname.lower().endswith(".jpg"):
                    continue
                path = os.path.join(directory, fname)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                faces.append(img)
                labels.append(student_id)

        if not faces:
            return False, 0, 0

        recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        recognizer.train(faces, np.array(labels))
        os.makedirs(self.trainer_dir, exist_ok=True)
        recognizer.save(self.model_path)
        self._recognizer = recognizer
        return True, len(faces), len(set(labels))

    def recognize(self, raw_image_bytes: bytes, confidence_threshold: float = 65.0) -> Dict:
        self._ensure_recognizer_loaded()
        if self._recognizer is None:
            return {"student_id": None, "confidence": None, "recognized": [], "message": "model not trained"}

        image_array = np.frombuffer(raw_image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame is None:
            return {"student_id": None, "confidence": None, "recognized": [], "message": "invalid image"}

        # Convert to grayscale and enhance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Better face detection parameters for recognition
        faces = self._face_detector.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(80, 80),
            maxSize=(400, 400)
        )
        
        if len(faces) == 0:
            return {"student_id": None, "confidence": None, "recognized": [], "message": "no face"}

        recognized_list: List[Dict] = []
        for (x, y, w, h) in faces:
            face_roi = gray[y : y + h, x : x + w]
            
            # Ensure minimum size
            if w < 80 or h < 80:
                continue
                
            face_resized = cv2.resize(face_roi, (200, 200))
            label, confidence = self._recognizer.predict(face_resized)
            
            # Lower confidence = better match in LBPH
            if confidence <= confidence_threshold:
                recognized_list.append({
                    "student_id": int(label),
                    "confidence": float(confidence),
                    "box": [int(x), int(y), int(w), int(h)],
                })

        # Backward-compat: keep top-1 in student_id/confidence
        top1 = min(recognized_list, key=lambda r: r["confidence"]) if recognized_list else None
        return {
            "student_id": (top1["student_id"] if top1 else None),
            "confidence": (top1["confidence"] if top1 else None),
            "recognized": recognized_list,
        }
