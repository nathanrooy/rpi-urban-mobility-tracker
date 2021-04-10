#--- IMPORT DEPENDENCIES ------------------------------------------------------+

import os
import time
import argparse

import cv2
import numpy as np

# deep sort
from deep_sort.tracker import Tracker
from deep_sort import nn_matching
 
# umt utils
from umt.umt_utils import parse_label_map
from umt.umt_utils import initialize_detector
from umt.umt_utils import initialize_img_source
from umt.umt_utils import generate_detections

#--- CONSTANTS ----------------------------------------------------------------+

LABEL_PATH = "models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/labelmap.txt"
DEFAULT_LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), LABEL_PATH)
TRACKER_OUTPUT_TEXT_FILE = 'object_paths.csv'

# deep sort related
MAX_COSINE_DIST = 0.4
NN_BUDGET = None
NMS_MAX_OVERLAP = 1.0

#--- MAIN ---------------------------------------------------------------------+

def main():
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
    parser.add_argument('-display', dest='live_view', required=False, default=False, action='store_true', help='add this flag to view a live display. note, that this will greatly slow down the fps rate.')
    parser.add_argument('-save', dest='save_frames', required=False, default=False, action='store_true', help='add this flag if you want to persist the image output. note, that this will greatly slow down the fps rate.')
    args = parser.parse_args()
    
    # basic checks
    if args.model_path: assert args.label_map_path, "when specifying a custom model, you must also specify a label map path using: '-labelmap <path to labelmap.txt>'"
    if args.model_path: assert os.path.exists(args.model_path)==True, "can't find the specified model..."
    if args.label_map_path: assert os.path.exists(args.label_map_path)==True, "can't find the specified label map..."
    if args.video_path: assert os.path.exists(args.video_path)==True, "can't find the specified video file..."

    print('> INITIALIZING UMT...')
    print('   > THRESHOLD:',args.threshold)

	# parse label map
    labels = parse_label_map(args, DEFAULT_LABEL_MAP_PATH)
    
    # initialize detector
    interpreter = initialize_detector(args)

    # create output directory
    if not os.path.exists('output') and args.save_frames: os.makedirs('output')
 
 	# initialize deep sort tracker   
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DIST, NN_BUDGET)
    tracker = Tracker(metric) 

    # initialize image source
    img_generator = initialize_img_source(args)

    # initialize plot colors (if necessary)
    if args.live_view or args.save_frames: COLORS = (np.random.rand(32, 3) * 255).astype(int)

    # main tracking loop
    print('\n> TRACKING...')
    with open(TRACKER_OUTPUT_TEXT_FILE, 'w') as out_file:
        for i, pil_img in enumerate(img_generator(args)):
        
            f_time = int(time.time())
            print('> FRAME:', i)
            
            # add header to trajectory file
            if i == 0:
            	header = (f'frame_num,rpi_time,obj_class,obj_id,obj_age,'
            	    'obj_t_since_last_update,obj_hits,'
            	    'xmin,ymin,xmax,ymax')
            	print(header, file=out_file)

            # get detections
            detections = generate_detections(pil_img, interpreter, args.threshold)
			
            # proceed to updating state
            if len(detections) == 0: print('   > no detections...')
            else:
            
                # update tracker
                tracker.predict()
                tracker.update(detections)
                
                # save object locations
                if len(tracker.tracks) > 0:
                    for track in tracker.tracks:
                        bbox = track.to_tlbr()
                        class_name = labels[track.class_name]
                        row = (f'{i},{f_time},{class_name},'
                            f'{track.track_id},{int(track.age)},'
                            f'{int(track.time_since_update)},{str(track.hits)},'
                            f'{int(bbox[0])},{int(bbox[1])},'
                            f'{int(bbox[2])},{int(bbox[3])}')
                        print(row, file=out_file)
                
            # only for live display
            if args.live_view or args.save_frames:
            
            	# convert pil image to cv2
                cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            	# cycle through actively tracked objects
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    
                    # draw detections and label
                    bbox = track.to_tlbr()
                    class_name = labels[track.class_name]
                    color = COLORS[int(track.track_id) % len(COLORS)].tolist()
                    cv2.rectangle(cv2_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(cv2_img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(str(class_name))+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(cv2_img, str(class_name) + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

                # live view
                if args.live_view:
                    cv2.imshow("tracker output", cv2_img)
                    cv2.waitKey(1)
                    
                # persist frames
                if args.save_frames: cv2.imwrite(f'output/frame_{i}.jpg', cv2_img)
                
    cv2.destroyAllWindows()         
    pass


#--- MAIN ---------------------------------------------------------------------+

if __name__ == '__main__':
    main()
     
#--- END ----------------------------------------------------------------------+
