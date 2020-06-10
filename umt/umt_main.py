#--- IMPORT DEPENDENCIES ------------------------------------------------------+

import os
import time
import argparse

from umt.sort import Sort

from umt.umt_utils import parse_label_map
from umt.umt_utils import initialize_detector
from umt.umt_utils import initialize_img_source
from umt.umt_utils import generate_detections
from umt.umt_utils import match_detections_to_labels_and_scores
from umt.umt_utils import persist_image_output
from umt.umt_utils import plot_colors

#--- CONSTANTS ----------------------------------------------------------------+

LABEL_PATH = "models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/labelmap.txt"
DEFAULT_LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), LABEL_PATH)
TRACKER_OUTPUT_TEXT_FILE = 'object_paths.txt'

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
    parser.add_argument('-display', dest='display', required=False, default=False, action='store_true', help='add this flag to output images from tracker. note, that this will greatly slow down the fps rate.')
    args = parser.parse_args()
    

    # basic checks
    if args.model_path: assert args.label_map_path, "when specifying a custom model, you must also specify a label map path using: '-labelmap <path to labelmap.txt>'"
    if args.model_path: assert os.path.exists(args.model_path)==True, "can't find the specified model..."
    if args.label_map_path: assert os.path.exists(args.label_map_path)==True, "can't find the specified label map..."
    if args.video_path: assert os.path.exists(args.video_path)==True, "can't find the specified video file..."


    print('> INITIALIZING UMT...')
    print('   > THRESHOLD:',args.threshold)

    # initialize detector
    interpreter = initialize_detector(args)
        
    # parse label map
    labels = parse_label_map(args, DEFAULT_LABEL_MAP_PATH)

    # create output directory
    if not os.path.exists('output'): os.makedirs('output')

    # create instance of the SORT tracker
    tracker = Sort()

    # initialize image source
    img_generator = initialize_img_source(args)

    # initialize plot colors (if necessary)
    if args.display: COLORS = plot_colors()

    # main tracking loop
    print('\n> TRACKING...')
    with open(TRACKER_OUTPUT_TEXT_FILE, 'w') as out_file:
        for i, pil_img in enumerate(img_generator(args)):

            f_time = int(time.time())
            print('> FRAME:', i)

            # get detections
            new_dets, classes, scores = generate_detections(pil_img, interpreter, args.threshold)

            # proceed to updating state
            if new_dets.shape[0] > 0:

                # sometimes the sort algo fails...
                try:
                    # update tracker
                    trackers = tracker.update(new_dets)
                
                    # match classes up to detections
                    tracker_labels, tracker_scores = match_detections_to_labels_and_scores(new_dets, trackers, scores, classes, labels)

                    # save image output
                    if(args.display): persist_image_output(pil_img, trackers, tracker_labels, tracker_scores, COLORS, i)
                    
                    # save object locations
                    for d, tracker_label, tracker_score in zip(trackers, tracker_labels, tracker_scores):
                        print(f'{i},{f_time},{d[4]},{d[0]},{d[1]},{d[2]-d[0]},{d[3]-d[1]},{tracker_label},{tracker_score}', file=out_file)

                except:
                    print('   > TRACKER FAILED...')
                    pass
     
    pass


#--- MAIN ---------------------------------------------------------------------+

if __name__ == '__main__':
    main()
    
#--- END ----------------------------------------------------------------------+
