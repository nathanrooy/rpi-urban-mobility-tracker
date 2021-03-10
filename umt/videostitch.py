# Script for generating a video when using umt with -save.
# Referenced a solution on StackOverflow by Ehsan

import os
os.system("ffmpeg -f image2 -i output/frame_%01d.jpg -vcodec mpeg4 -y output-stitched.mp4")

