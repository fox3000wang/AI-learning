import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# Call Detection Models

# Method-1, use FaceAnalysis
# enable detection model only
app = FaceAnalysis(allowed_modules=['detection'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Method-2, load model directly
detector = insightface.model_zoo.get_model('your_detection_model.onnx')
detector.prepare(ctx_id=0, input_size=(640, 640))


# Call Recognition Models
handler = insightface.model_zoo.get_model('your_recognition_model.onnx')
handler.prepare(ctx_id=0)
