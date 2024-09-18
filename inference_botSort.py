import os
import time
from typing import Tuple
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from tracker.bot_sort import BoTSORT
import argparse

print("TensorRT version:", trt.__version__)

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)[1:]) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        binding_index = engine.get_binding_index(binding)

        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))

        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path), f"Engine file does not exist: {engine_file_path}"
    print(f"Reading engine from file {engine_file_path}")
    trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.INFO), "")
    with open(engine_file_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.INFO)) as runtime:
        serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        return engine

class InferenceSession:
    def __init__(self, engine_file, inference_shape: Tuple[int, int]):
        self.engine = load_engine(engine_file)
        self.context = None
        self.inference_shape = inference_shape

    def initialize(self):
        self.context = self.engine.create_execution_context()
        if not self.context:
            raise RuntimeError("Failed to create TensorRT execution context")

        # Print binding information
        for binding in self.engine:
            print("Binding", binding)
            print("Shape", self.engine.get_binding_shape(binding))
            print("DType", self.engine.get_binding_dtype(binding))

        input_binding_index = self.engine.get_binding_index("input")
        self.context.set_binding_shape(input_binding_index, (1, 3, *self.inference_shape))

        # Allocate buffers correctly for input and output
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)

    def __enter__(self):
        self.initialize()
        return self

    def preprocess(self, image):
        image = np.array(image)
        rows, cols = self.inference_shape
        original_shape = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(cols, rows))
        return np.moveaxis(image, 2, 0), original_shape

    def postprocess(self, detected_boxes, original_shape: Tuple[int, int]):
        sx = original_shape[1] / self.inference_shape[1]
        sy = original_shape[0] / self.inference_shape[0]
        detected_boxes[:, [0, 2]] *= sx
        detected_boxes[:, [1, 3]] *= sy
        return detected_boxes

    def __call__(self, image):
        batch_size = 1
        input_image, original_shape = self.preprocess(image)

        self.inputs[0].host[:np.prod(input_image.shape)] = np.asarray(input_image).ravel()

        [cuda.memcpy_htod(inp.device, inp.host) for inp in self.inputs]

        success = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        assert success
        self.stream.synchronize()
        [cuda.memcpy_dtoh(out.host, out.device) for out in self.outputs]

        num_detections, detected_boxes, detected_scores, detected_labels = [o.host for o in self.outputs]

        num_detections = int(num_detections[0])

        if num_detections == 0:
            # If no detections, return empty arrays
            return num_detections, np.array([]), np.array([]), np.array([])

        detected_boxes = detected_boxes.reshape(-1, 4)[:num_detections]
        detected_scores = detected_scores.reshape(-1)[:num_detections]
        detected_labels = detected_labels.reshape(-1)[:num_detections]

        detected_boxes = self.postprocess(detected_boxes, original_shape)
        
        # Debug information
        print(f"Number of detections: {num_detections}")
        print(f"Boxes shape: {detected_boxes.shape}")
        print(f"Scores shape: {detected_scores.shape}")
        print(f"Labels shape: {detected_labels.shape}")
        print(f"Raw detected boxes: {detected_boxes}")
        print(f"Raw detected scores: {detected_scores}")
        print(f"Raw detected labels: {detected_labels}")

        return num_detections, detected_boxes, detected_scores, detected_labels

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.inputs, self.outputs, self.bindings, self.stream, self.context

if __name__ == "__main__":
    engine_path = "model.trt"
    inference_shape = (640, 640)

    try:
        session = InferenceSession(engine_path, inference_shape)
        session.initialize()
    except RuntimeError as e:
        print(f"Error during initialization: {e}")
        exit(1)

    cap = cv2.VideoCapture(".")
    if not cap.isOpened():
        print("Error: Unable to open input video")
        exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height))

    total_frames = 0
    total_time = 0

    botsort_args = {
        "mot20":False,
        "match_thresh":0.8,
        "track_high_thresh": 0.5,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.6,

        "max_age": 30,
        "min_box_area": 10,
        "fuse_score":True,
    
        "track_buffer": 30,
        "proximity_thresh": 0.5,
        "appearance_thresh": 0.25,
        "with_reid": False,
        "cmc_method": "sparseOptFlow",
        "name": "BoTSORT",
        "ablation": False 
       
    }

    # Convert dictionary to Namespace for BoTSORT
    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    botsort_args = Namespace(**botsort_args)
    tracker = BoTSORT(args=botsort_args)

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame")
            break

        try:
            # Run inference
            num_detections, detected_boxes, detected_scores, detected_labels = session(frame)

            detections = []
            for i in range(num_detections):
                if detected_scores[i] > 0.75:  # Filter based on confidence threshold
                    box = detected_boxes[i]
                    x1, y1, x2, y2 = map(int, box)
                    cls = detected_labels[i]
                    conf = detected_scores[i]
                    #if cls == 2:  
                    detections.append([x1, y1, x2, y2, conf, 1])  # Format: [x1, y1, x2, y2, score, class_id]

            if detections:
                detections_np = np.array(detections)  # Convert to NumPy array
                print("Detections for tracking:", detections_np)

                if frame.ndim == 2:  
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                try:
                   
                    track_info = tracker.update(detections_np, frame)

                    for track in track_info:
                        track_id = track.track_id
                        x1, y1, x2, y2 = track.tlbr.astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                except Exception as e:
                    print(f"Error during tracking update: {e}")

            end_time = time.time()
            processing_time = end_time - start_time
            total_time += processing_time
            total_frames += 1

            fps = total_frames / total_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out.write(frame)

        except Exception as e:
            print(f"Error during frame processing: {e}")

    cap.release()
    out.release()

    avg_fps = total_frames / total_time if total_time > 0 else 0
    print(f"Average FPS: {avg_fps:.2f}")
