# Script for generating a video when using umt with -save.
# Referenced a solution on StackOverflow by BoboDarph

import cv2
import os

image_folder = 'output'
video_name = 'stitch.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()