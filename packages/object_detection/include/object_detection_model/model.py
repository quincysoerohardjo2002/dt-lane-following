#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import onnxruntime as ort
from dt_computer_vision.camera.types import Pixel

from object_detection_model.config import MODEL_PATH, CONF_THRESHOLD, STOP_DISTANCE, FORWARD_PWM

class MLModel:
    def __init__(self):
        print("Initializing MLModel")
        self.ground_projector = None

        if not MODEL_PATH.exists():
            raise FileNotFoundError("ONNX model not found (did you download your trained model?):", MODEL_PATH)

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=sess_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"], 
        )

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.in_dtype = np.float16 if inp.type == "tensor(float16)" else np.float32

        self.net_h = inp.shape[2]
        self.net_w = inp.shape[3]


    def _run_detector(self, img_bgr):
        x = self._preprocess(img_bgr)
        out = self.session.run(None, {self.input_name: x})[0]  # shape [1,N,6]
        return out[0]


    def _should_stop(self, detections: np.ndarray):
        if self.ground_projector is None:
            print("ground_projector is None")
            return False
    
        for x1, y1, x2, y2, score, _ in detections:
            print(f"\nDetection: x1={x1}, y1={y1}, x2={x2}, y2={y2}, score={score}")
    
            if score < CONF_THRESHOLD:
                print("Skipped: low confidence")
                continue
            
            u = int((x1 + x2) / 2)
            v = int(y2)
    
            print(f"Bottom-center pixel: u={u}, v={v}")
    
            try:
                pix = Pixel(x=u, y=v)
                vec = self.ground_projector.camera.pixel2vector(pix)
                ground_point = self.ground_projector.vector2ground(vec)
    
                dist = np.sqrt(ground_point.x**2 + ground_point.y**2)
    
                print(f"Ground point: x={ground_point.x}, y={ground_point.y}")
                print(f"Distance: {dist}, STOP_DISTANCE={STOP_DISTANCE}")
    
                if ground_point.x > 0 and dist < STOP_DISTANCE:
                    print("STOP CONDITION MET")
                    return True
    
            except Exception as e:
                print(f"Projection error: {e}")
                continue
            
        print("No stop condition met")
        return False


    def _preprocess(self, img_bgr):
        h, w = img_bgr.shape[:2]

        if h != self.net_h or w != self.net_w:
            raise ValueError(
                f"Image size {h}x{w} does not match ONNX! Expected {self.net_h}x{self.net_w}"
            )

        img = img_bgr[:, :, ::-1].astype(self.in_dtype) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return img


    def set_ground_projector(self, gp):
        self.ground_projector = gp
        

    def get_wheel_velocities_from_image(self, img: np.ndarray):
        try:
            detections = self._run_detector(img)
        except Exception as e:
            print(f"ONNX inference error {e}")
            return [None, None, True]
        should_stop = self._should_stop(detections)
        return [detections, should_stop, False]
