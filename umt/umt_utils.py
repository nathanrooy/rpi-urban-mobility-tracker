import os
from time import sleep
import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np

import cv2
from scipy.spatial.distance import cosine

import imutils
from imutils.video import VideoStream

# deep sort
#from umt.deep_sort import generate_detections as gd
#from umt.deep_sort.detection import Detection
#from umt.deep_sort.preprocessing import non_max_suppression
from deep_sort.detection import Detection
from deep_sort_tools import generate_detections as gd


# constants
nms_max_overlap = 1.0

# initialize an instance of the deep-sort tracker
w_path = os.path.join(os.path.dirname(__file__), 'deep_sort/mars-small128.pb')
encoder = gd.create_box_encoder(w_path, batch_size=1)
    

def camera_frame_gen(args):

    # initialize the video stream and allow the camera sensor to warmup
    print("> starting video stream...")
    vs = VideoStream(src=0).start()
    sleep(2.0)

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
    
    # keep detections of specified classes
    #
    #
	#...
	

    # denormalize bounding box dimensions
    if len(keep_idx) > 0:
        bboxes[:,0] = bboxes[:,0] * pil_img_obj.size[1]
        bboxes[:,1] = bboxes[:,1] * pil_img_obj.size[0]
        bboxes[:,2] = bboxes[:,2] * pil_img_obj.size[1]
        bboxes[:,3] = bboxes[:,3] * pil_img_obj.size[0]
    
    # convert bboxes from [ymin, xmin, ymax, xmax] -> [xmin, ymin, width, height]
    for box in bboxes:
        xmin = int(box[1])
        ymin = int(box[0])
        w = int(box[3]) - xmin
        h = int(box[2]) - ymin
        box[0], box[1], box[2], box[3] = xmin, ymin, w, h
		
    # generate features for deepsort
    features = encoder(np.array(pil_img_obj), bboxes)

    # munge into deep sort detection objects
    detections = [Detection(bbox, score, feature, class_name) for bbox, score, feature, class_name in zip(bboxes, scores, features, classes)]

    # run non-maximum suppression
    # borrowed from: https://github.com/nwojke/deep_sort/blob/master/deep_sort_app.py#L174
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = non_max_suppression(boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]
    
    return detections


def parse_label_map(args, DEFAULT_LABEL_MAP_PATH):
    if args.label_map_path == DEFAULT_LABEL_MAP_PATH: print('   > CUSTOM LABEL MAP = FALSE')
    else: print(f'   > CUSTOM LABEL MAP = TRUE ({args.label_map_path})')

    labels = {}
    for i, row in enumerate(open(args.label_map_path)):
        labels[i] = row.replace('\n','')
    return labels


def non_max_suppression(boxes, max_bbox_overlap, scores=None):
    """Suppress overlapping detections.
    Original code from [1]_ has been adapted to include confidence score.
    .. [1] http://www.pyimagesearch.com/2015/02/16/
           faster-non-maximum-suppression-python/
    Examples
    --------
        >>> boxes = [d.roi for d in detections]
        >>> scores = [d.confidence for d in detections]
        >>> indices = non_max_suppression(boxes, max_bbox_overlap, scores)
        >>> detections = [detections[i] for i in indices]
    Parameters
    ----------
    boxes : ndarray
        Array of ROIs (x, y, width, height).
    max_bbox_overlap : float
        ROIs that overlap more than this values are suppressed.
    scores : Optional[array_like]
        Detector confidence score.
    Returns
    -------
    List[int]
        Returns indices of detections that have survived non-maxima suppression.
    """
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + boxes[:, 0]
    y2 = boxes[:, 3] + boxes[:, 1]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if scores is not None:
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(
            idxs, np.concatenate(
                ([last], np.where(overlap > max_bbox_overlap)[0])))

    return pick
