import json
from PIL import Image
import numpy as np
from os import path
import math

def readJson(jsonPath):
	with open(jsonPath) as f:
		imageInfos = json.load(f)

	allImages = []
	ln_ts = []

	# read all images and store as numpy.array
	for imageInfo in imageInfos:
		image = Image.open(
			path.join(path.dirname(jsonPath), imageInfo["path"]))
		allImages.append(np.array(image))
		ln_ts.append(math.log(imageInfo["t"], math.e))

	allImages = np.array(allImages)
	ln_ts = np.array(ln_ts)

	return allImages, ln_ts