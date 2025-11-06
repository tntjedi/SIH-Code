# detect_onnx.py
import cv2
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (640, 640))
    img = img[:, :, ::-1] / 255.0
    img = img.transpose(2, 0, 1).astype(np.float32)
    img = np.expand_dims(img, axis=0)

    outputs = session.run(None, {input_name: img})[0]

    # NOTE: Here you must post-process outputs (NMS, scaling boxes back).
    # If you want, I can generate your exact post-processing code.
    
    cv2.imshow("Inference (ONNX)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()