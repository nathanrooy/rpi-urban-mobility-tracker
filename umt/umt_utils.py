import os
import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np

from umt.sort import *

import cv2
from scipy.spatial.distance import cosine

import imutils
from imutils.video import VideoStream

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_colors():
    return np.random.rand(32, 3)


def camera_frame_gen(args):

    # initialize the video stream and allow the camera sensor to warmup
    print("> starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # pull frame from video stream
        frame = vs.read()

        # array to PIL image format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield Image.fromarray(frame)

    pass


def image_seq_gen(args):

    # collect images to be processed
    images = []
    for item in sorted(os.listdir(args.image_path)):
        if item[-4:] == '.jpg': images.append(f'{args.image_path}{item}')
    
    # cycle through image sequence and yield a PIL img object
    for frame in range(0, args.nframes): yield Image.open(images[frame])


def video_frame_gen(args):
    
    counter = 0
    cap = cv2.VideoCapture(args.video_path)
    while(cap.isOpened()):
        counter += 1
        if counter > args.nframes: break
        if cv2.waitKey(1) & 0xFF == ord('q'): break

        # pull frame from video stream
        _, frame = cap.read()

        # array to PIL image format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        yield Image.fromarray(frame)


def persist_image_output(pil_img, trackers, tracker_labels, tracker_scores, COLORS, frame):

    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_axis_off()
    fig.add_axes(ax)

    plt.imshow(pil_img)
    ax.set_aspect('equal')

    for d, label, score in zip(trackers, tracker_labels, tracker_scores):
        d = d.astype(np.int32)
        plt.text(d[0], d[1] - 20, f'{label} (#{d[4]%32})', color=COLORS[d[4]%32,:], fontsize=10, fontweight='bold')
        plt.text(d[0], d[1] -  5, f'{score:0.4f}', color=COLORS[d[4]%32,:], fontsize=10, fontweight='bold')
        rect = Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1], fill=False, lw=2, ec=COLORS[d[4]%32,:])
        ax.add_patch(rect)
        
    plt.savefig(f'output/frame_{frame}.jpg', bbox_inches='tight', pad_inches=0)

    pass


def initialize_img_source(args):

    # track objects from video file
    if args.video_path: return video_frame_gen
    
    # track objects in image sequence
    if args.image_path: return image_seq_gen
        
    # track objects from camera source
    if args.camera: return camera_frame_gen


def initialize_detector(args):

    TPU_PATH = 'models/tpu/mobilenet_ssd_v2_coco_quant/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    CPU_PATH = 'models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/detect.tflite'

    # initialize coral tpu model
    if args.tpu:
        print('   > TPU = TRUE')
        
        if args.model_path:
            model_path = args.model_path
            print('   > CUSTOM DETECTOR = TRUE')
            print(f'      > DETECTOR PATH = {model_path}')
        	
        else:
        	model_path = os.path.join(os.path.dirname(__file__), TPU_PATH)
        	print('   > CUSTOM DETECTOR = FALSE')
        
        _, *device = model_path.split('@')
        edgetpu_shared_lib = 'libedgetpu.so.1'
        interpreter = tflite.Interpreter(
                model_path,
                experimental_delegates=[
                    tflite.load_delegate(edgetpu_shared_lib,
                        {'device': device[0]} if device else {})
                ])
        interpreter.allocate_tensors()

    # initialize tflite model
    else:
        print('   > TPU = FALSE')
        
        if args.model_path:
            model_path = args.model_path
            print('   > CUSTOM DETECTOR = TRUE')
            print(f'      > DETECTOR PATH = {model_path}')
        	
        else:
        	print('   > CUSTOM DETECTOR = FALSE')
        	model_path = os.path.join(os.path.dirname(__file__), CPU_PATH)
        
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

    return interpreter


def match_detections_to_labels_and_scores(detections, trackers, scores, classes, labels):
    
    iou_matrix = np.zeros((len(trackers), len(detections)),dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_area = iou(det,trk)
            if np.isnan(iou_area)==False: iou_matrix[t,d]=iou_area
            else: iou_matrix[t,d]=0

    matched_indices = np.array(linear_assignment(-iou_matrix)).T

    matched_classes = classes[matched_indices[:,1]]
    matched_labels = [labels[item] for item in matched_classes]
    matched_scores = scores[matched_indices[:,1]]

    return matched_labels, matched_scores


def generate_detections(pil_img_obj, interpreter, threshold):
    
    # resize image to match model input dimensions
    img = pil_img_obj.resize((interpreter.get_input_details()[0]['shape'][2], 
                              interpreter.get_input_details()[0]['shape'][1]))

    # add n dim
    input_data = np.expand_dims(img, axis=0)

    # infer image
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()

    # collect results
    bboxes = np.squeeze(interpreter.get_tensor(interpreter.get_output_details()[0]['index']))
    classes = np.squeeze(interpreter.get_tensor(interpreter.get_output_details()[1]['index']) + 1).astype(np.int32)
    scores = np.squeeze(interpreter.get_tensor(interpreter.get_output_details()[2]['index']))
    num = np.squeeze(interpreter.get_tensor(interpreter.get_output_details()[3]['index']))
    
    # keep detections above specified threshold
    keep_idx = np.less(scores[np.greater(scores, threshold)], 1)
    bboxes  = bboxes[:keep_idx.shape[0]][keep_idx]
    classes = classes[:keep_idx.shape[0]][keep_idx]
    scores = scores[:keep_idx.shape[0]][keep_idx]

    # denormalize bounding box dimensions
    if len(keep_idx) > 0:
        bboxes[:,0] = bboxes[:,0] * pil_img_obj.size[1]
        bboxes[:,1] = bboxes[:,1] * pil_img_obj.size[0]
        bboxes[:,2] = bboxes[:,2] * pil_img_obj.size[1]
        bboxes[:,3] = bboxes[:,3] * pil_img_obj.size[0]
        
        return np.hstack((bboxes[:,[1,0,3,2]], np.full((bboxes.shape[0], 1), 50))).astype(np.int16), classes, scores
    else: return np.array([]), np.array([]), np.array([])


def parse_label_map(args, DEFAULT_LABEL_MAP_PATH):
    if args.label_map_path == DEFAULT_LABEL_MAP_PATH: print('   > CUSTOM LABEL MAP = FALSE')
    else: print(f'   > CUSTOM LABEL MAP = TRUE ({args.label_map_path})')

    labels = {}
    for i, row in enumerate(open(args.label_map_path)):
        labels[i] = row.replace('\n','')
    return labels
