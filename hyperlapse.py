__author__ = "Sam Prestwood"
__email__ = "swp2sf@virginia.edu"

"""
hyperlapse.py
Hyperlapse final project
Computational Photography, Fall 2014
"""

# imports:

import numpy
import matplotlib.pyplot as plt
import skimage.io
import cv2
import cv2.cv as cv
from savitzky_golay_filter import savgol
import sys
import copy

# globals:

INPUT_VIDEO_PATH = "path/to/input/file"
OUTPUT_VIDEO_PATH = "path/to/output/file"
SPEED_UP = 15
CORNER_HARRIS_K = 0.04 
CORNER_HARRIS_K_SIZE = 3
CORNER_HARRIS_BLOCK_SIZE = 2
CORNER_THRESH = 0.05
LK_WINDOW_SIZE = (30, 30)
LK_MAX_LEVEL = 4
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001)
GF_MAX_CORNERS = 100
GF_QUALITY_LEVEL = 0.3
GF_MIN_DISTANCE = 7
GF_BLOCK_SIZE = 7
FOURCC = cv.CV_FOURCC(*'XVID')
SG_WINDOW = 359
SG_P_ORDER = 3
CROP_THRESH = 0.215

# functions:

def find_motion_vector(img1, img2):
	"""given 2 contiguous images in an image stream, returns the motion vector \
	 between them"""

	if img1 == None or img2 == None or img1.shape[2] < 3 or img2.shape[2] < 3:
		return numpy.array([0.0, 0.0])

	gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
	features1 = cv2.goodFeaturesToTrack(gray1, GF_MAX_CORNERS, \
		GF_QUALITY_LEVEL, GF_MIN_DISTANCE, GF_BLOCK_SIZE)

	if features1 == None or len(features1) == 0:
		return numpy.array([0.0, 0.0])

	gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
	features2, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, features1, \
		nextPts=None, winSize=LK_WINDOW_SIZE, maxLevel=LK_MAX_LEVEL, \
		criteria=LK_CRITERIA)

	good_features2 = features2[st==1]
	good_features1 = features1[st==1]

	diff = good_features1 - good_features2

	# if no correspondences are found:
	if len(diff) == 0: 
		return numpy.array([0.0, 0.0])

	return numpy.mean(diff, axis=0, dtype=numpy.float32)

def pack_for_csv(list_of_vals):
	"""returns a csv string of the vals in the list"""

	string = ""

	for val in list_of_vals:
		string += (str(val) + ",")

	return string

def write_progress(p):
	"""writes to the console the progress that something is at"""
	sys.stdout.write("\r")
	sys.stdout.write(str(p))
	sys.stdout.flush()

def better_stabilization(input_file, output_file):
	"""Smooths the video rather than fully stabilizing it"""
	# http://docs.opencv.org/trunk/doc/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

	global GF_MAX_CORNERS, GF_QUALITY_LEVEL, GF_QUALITY_LEVEL, GF_MIN_DISTANCE,\
		FOURCC

	video = cv2.VideoCapture(input_file)

	width = int(video.get(cv.CV_CAP_PROP_FRAME_WIDTH))
	height = int(video.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
	fps = video.get(cv.CV_CAP_PROP_FPS)
	total_frames = int(video.get(cv.CV_CAP_PROP_FRAME_COUNT))

	x = 0
	y = 0

	pos = []
	pos.append([x, y])

	print "computing motion between frames..."

	returnVal, frame1 = video.read()

	for i in range(1, total_frames):
		returnVal, frame2 = video.read()
		vec = find_motion_vector(frame1, frame2)

		x += vec[0]
		y += vec[1]

		pos.append([x, y])

		frame1 = frame2
		write_progress(str(round(i / float(total_frames), 3) * 100) + \
			"% complete")

	pos = numpy.array(pos)
	video.release()

	print "\nsmoothing motion...",
	
	window = int(6 * fps)
	if window % 2 == 0:
		window += 1
	print "(using window size of " + str(window) + ")"

	smoothed = numpy.copy(pos)
	smoothed[:, 0] = savgol(pos[:, 0], window, SG_P_ORDER)
	smoothed[:, 1] = savgol(pos[:, 1], window, SG_P_ORDER)

	print "subsampling frames..."

	pos = pos[::SPEED_UP]
	smoothed = smoothed[::SPEED_UP]
	cumulative_motion = copy.deepcopy(smoothed)

	print len(cumulative_motion), len(pos), len(smoothed)
	if len(cumulative_motion) != len(pos):
		print "error! lengths don't equal!"
		exit()

	print "smoothing subsampled motion..."

	window = int(6 * fps)
	if window % 2 == 0:
		window += 1
	print "(using window size of " + str(window) + ")"

	cumulative_motion[:, 0] = savgol(cumulative_motion[:, 0], window, SG_P_ORDER)
	cumulative_motion[:, 1] = savgol(cumulative_motion[:, 1], window, SG_P_ORDER)

	trans = pos - cumulative_motion

	'''
	# plots motion data:

	t = []
	for i in range(len(pos)):
		t.append(i)
	
	plt.plot(t, pos[:, 0], 'ro',
			 t, pos[:, 1], 'r^',
			 t, smoothed[:, 0], 'go',
			 t, smoothed[:, 1], 'g^',
			 t, trans[:, 0], 'mo',
			 t, trans[:, 1], 'm^',
			 t, cumulative_motion[:, 0], 'bo',
			 t, cumulative_motion[:, 1], 'b^')
	plt.show()
	'''

	print "translating frames..."

	i = 0

	video = cv2.VideoCapture(input_file)

	x_max, y_max = trans.max(axis = 0)
	x_min, y_min = trans.min(axis = 0)
	x_crop = max(abs(x_max), abs(x_min))
	y_crop = max(abs(y_max), abs(y_min))
	
	print "x_crop =", x_crop, "(", str(round(x_crop * 100. / width, 3)), 
	print "% of width),", "y_crop =", y_crop, "(", 
	print str(round(y_crop * 100. / height, 3)), "% of height)" 
	
	if x_crop / width > CROP_THRESH or y_crop / height > CROP_THRESH:
		print "Too much video to crop, doing frame overlay instead"

		writer = cv2.VideoWriter(output_file, FOURCC, fps, (width, height))
		returnVal, frame = video.read()
		canvas =  numpy.zeros((height, width, 3), 'uint8')

		for i in range(len(trans)):
			m = numpy.float32([[1, 0, trans[i][0]], [0, 1, trans[i][1]]])
			dst = cv2.warpAffine(frame, m, (width, height))

			matte = (numpy.clip(dst, 0, 1) - 1) * (-1)
			canvas = canvas * matte + dst

			writer.write(canvas.astype(numpy.uint8))
			write_progress(str(round(i / float(len(trans)), 3) * 100) + \
				"% complete")

			for j in range(SPEED_UP):
				returnVal, frame = video.read()
				if not returnVal:
					break

	else:
		print "Cropping frame:",
		print "(" + str(width) + ", "  + str(height) + ") ->",
		width = int(width - (2 * x_crop))
		height = int(height - (2 * y_crop))
		print "(" + str(width) + ", "  + str(height) + ")"

		writer = cv2.VideoWriter(output_file, FOURCC, fps, (width, height))
		returnVal, frame = video.read()

		for i in range(len(trans)):
			m = numpy.float32([[1, 0, trans[i][0] - x_crop], 
							   [0, 1, trans[i][1] - y_crop]])
			dst = cv2.warpAffine(frame, m, (width, height))
			
			writer.write(dst)
			write_progress(str(round(i / float(len(trans)), 3) * 100) + \
				"% complete")

			for j in range(SPEED_UP):
				returnVal, frame = video.read()
				if not returnVal:
					break

	print "\n...done!"

	video.release()
	writer.release()
	cv2.destroyAllWindows()

# main:

if __name__ == "__main__":
	better_stabilization(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH)