#--- IMPORT DEPENDENCIES ------------------------------------------------------+

import time
import argparse
from sort import *

import cv2
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine

import imutils
from imutils.video import VideoStream

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import tflite_runtime.interpreter as tflite

#--- CONSTANTS ----------------------------------------------------------------+

DEFAULT_LABEL_MAP_PATH = "models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/labelmap.txt"

#--- FUNCTIONS ----------------------------------------------------------------+


def parse_label_map(agrs):
    if args.label_map_path == DEFAULT_LABEL_MAP_PATH: print('   > CUSTOM LABEL MAP = FALSE')
    else: print(f'   > CUSTOM LABEL MAP = TRUE ({args.label_map_path})')

    labels = {}
    for i, row in enumerate(open(args.label_map_path)):
        labels[i] = row.replace('\n','')
    return labels


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
    keep_idx = np.greater(scores, threshold)
    bboxes  = bboxes[keep_idx]
    classes = classes[keep_idx]
    scores = scores[keep_idx]

    # denormalize bounding box dimensions
    if len(keep_idx) > 0:
        bboxes[:,0] = bboxes[:,0] * pil_img_obj.size[1]
        bboxes[:,1] = bboxes[:,1] * pil_img_obj.size[0]
        bboxes[:,2] = bboxes[:,2] * pil_img_obj.size[1]
        bboxes[:,3] = bboxes[:,3] * pil_img_obj.size[0]
        
        return np.hstack((bboxes[:,[1,0,3,2]], np.full((bboxes.shape[0], 1), 50))).astype(np.int16), classes, scores
    else: return []

    
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


def persist_image_output(pil_img, trackers, tracker_labels, tracker_scores, colors, frame):

    fig, ax = plt.subplots(figsize=(10,10))
    plt.imshow(pil_img)
    ax.set_aspect('equal')
    ax.set_title(' Tracked Targets')

    for d, label, score in zip(trackers, tracker_labels, tracker_scores):
        d = d.astype(np.int32)
        plt.text(d[0], d[1] - 20, f'{label} (#{d[4]%32})', color=colors[d[4]%32,:], fontsize=10, fontweight='bold')
        plt.text(d[0], d[1] -  5, f'{score:0.4f}', color=colors[d[4]%32,:], fontsize=10, fontweight='bold')
        rect = Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1], fill=False, lw=2, ec=colors[d[4]%32,:])
        ax.add_patch(rect)
        
    plt.savefig(f'output/frame_{frame}.jpg')

    pass
    
    
def track_image_seq(args, interpreter, tracker, labels, colors):

    # collect images to be processed
    images = []
    for item in sorted(os.listdir(args.image_path)):
        if '.jpg' in item:
            images.append(f'{args.image_path}{item}')
    
    # cycle through image sequence
    with open('object_paths.txt', 'w') as out_file:
        for frame in range(1, args.nframes, 1):

            print(f'> FRAME: {frame}')

            # load image
            pil_img = Image.open(images[frame])

            # get detections
            new_dets, classes, scores = generate_detections(pil_img, interpreter, args.threshold)

            # update tracker
            trackers = tracker.update(new_dets)
        
            # match classes up to detections
            tracker_labels, tracker_scores = match_detections_to_labels_and_scores(new_dets, trackers, scores, classes, labels)

            # save image output
            if(args.display):
                persist_image_output(pil_img, trackers, tracker_labels, tracker_scores, colors, frame)
            
            # save object locations
            for d, tracker_label, tracker_score in zip(trackers, tracker_labels, tracker_scores):
                print(f'{frame},{d[4]},{d[0]},{d[1]},{d[2]-d[0]},{d[3]-d[1]},{tracker_label},{tracker_score}', file=out_file)

    pass
    
    
def track_video(args, interpreter, tracker, labels, colors):

    assert os.path.exists(args.video_path)==True, "can't find the specified video file..."
    
    with open('object_paths.txt', 'w') as out_file:
    
        counter = 0
        cap = cv2.VideoCapture(args.video_path)
        while(cap.isOpened()):
            print(counter)

            # pull frame from video stream
            ret, frame = cap.read()

            # array to PIL image format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)

            # get detections
            new_dets, classes, scores = generate_detections(pil_img, interpreter, args.threshold)

            # update tracker
            trackers = tracker.update(new_dets)
       
            # match classes up to detections
            tracker_labels, tracker_scores = match_detections_to_labels_and_scores(new_dets, trackers, scores, classes, labels)

            # save image output
            if(args.display):
                persist_image_output(pil_img, trackers, tracker_labels, tracker_scores, colors, counter)

            # save object locations
            for d, tracker_label, tracker_score in zip(trackers, tracker_labels, tracker_scores):
                print(f'{counter},{d[4]},{d[0]},{d[1]},{d[2]-d[0]},{d[3]-d[1]},{tracker_label},{tracker_score}', file=out_file)

            counter += 1
            
            if counter >= args.nframes: break
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    pass
    
    
def track_camera(args, interpreter, tracker, labels, colors):

    # initialize the video stream and allow the camera sensor to warmup
    print("> starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    with open('object_paths.txt', 'w') as out_file:
        counter = 0
        while True:
            print(counter)

            # pull frame from video stream
            frame = vs.read()
            f_time = int(time.time())

            # array to PIL image format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)

            # get detections
            new_dets, classes, scores = generate_detections(pil_img, interpreter, args.threshold)

            # update tracker
            trackers = tracker.update(new_dets)
       
            # match classes up to detections
            tracker_labels, tracker_scores = match_detections_to_labels_and_scores(new_dets, trackers, scores, classes, labels)

            # save image output
            if(args.display):
                persist_image_output(pil_img, trackers, tracker_labels, tracker_scores, colors, counter)

            # save object locations
            for d, tracker_label, tracker_score in zip(trackers, tracker_labels, tracker_scores):
                print(f'{counter},{f_time},{d[4]},{d[0]},{d[1]},{d[2]-d[0]},{d[3]-d[1]},{tracker_label},{tracker_score}', file=out_file)

            counter += 1
    pass
    


def initialize_detector(args):

    # initialize coral tpu model
    if args.tpu:
        print('   > TPU = TRUE')
        
        if args.model_path:
            model_path = args.model_path
            print('   > CUSTOM DETECTOR = TRUE')
            print(f'      > DETECTOR PATH = {model_path}')
        	
        else:
        	model_path = 'models/tpu/mobilenet_ssd_v2_coco_quant/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
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
        	model_path = 'models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/detect.tflite'
        
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

    return interpreter
    

def main(args):
    print('> INITIALIZING UMT...')

    # initialize detector
    interpreter = initialize_detector(args)
        
    # parse label map
    if args.label_map_path: labels = parse_label_map(args.label_map_path)
    else: labels = parse_label_map("models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/labelmap.txt")
    
    # display only
    colors = np.random.rand(32, 3)

    # create output directory
    if not os.path.exists('output'):
        os.makedirs('output')

    # create instance of the SORT tracker
    tracker = Sort()

    print('\n> TRACKING...')
    
    # track objects from video file
    if args.video_path:
        track_video(args, interpreter, tracker, labels, colors)
    
    # track objects in image sequence
    if args.image_path:
        track_image_seq(args, interpreter, tracker, labels, colors)
        
    # track objects from camera source
    if args.camera:
        track_camera(args, interpreter, tracker, labels, colors)
     
    pass


#--- MAIN ---------------------------------------------------------------------+

if __name__ == '__main__':
    
    # parse arguments
    parser = argparse.ArgumentParser(description='--- Raspbery Pi Urban Mobility Tracker ---')
    parser.add_argument('-modelpath', dest='model_path', type=str, required=False, help='specify path of a custom detection model')
    parser.add_argument('-labelmap', dest='label_map_path', default=DEFAULT_LABEL_MAP_PATH, type=str, required=False, help='specify the label map text file')
    parser.add_argument('-imageseq', dest='image_path', type=str, required=False, help='specify an image sequence')
    parser.add_argument('-video', dest='video_path', type=str, required=False, help='specify video file')
    parser.add_argument('-camera', dest='camera', default=False, action='store_true', help='specify this when using the rpi camera as the input')
    parser.add_argument('-threshold', dest='threshold', type=float, default=0.5, required=False, help='specify a custom inference threshold')
    parser.add_argument('-tpu', dest='tpu', required=False, default=False, action='store_true', help='add this when using a coral usb accelerator')
    parser.add_argument('-nframes', dest='nframes', type=int, required=False, default=10, help='specify nunber of frames to process')
    parser.add_argument('-display', dest='display', required=False, default=False, action='store_true', help='add this flag to output images from tracker. note, that this will greatly slow down the fps rate.')
    args = parser.parse_args()

    # basic checks
    if args.model_path: assert args.label_map_path, "when specifying a custom model, you must also specify a label map path using: '-labelmap <path to labelmap.txt>'"
    
    # begin tracking
    main(args)
    
#--- END ----------------------------------------------------------------------+
