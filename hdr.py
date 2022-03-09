import numpy as np
from numpy.linalg import lstsq
from numpy.random import randint

from PIL import Image

import matplotlib.pyplot as plt

from numba import jit

import json
from os import path
import math

import argparse

# imagesInfo:[
# 	{
#		"path": "path/to/image",
#		"t": 0.3
#	}
# ]

@jit(nopython=True)
def getLnEnergy(allImages, g_Z, ln_ts, channel):
	energy = np.zeros(allImages[0].shape[:2], dtype='float64')
	for i in range(energy.shape[0]):
		for j in range(energy.shape[1]):
			
			sum = 0
			weight_sum = 0
			for k in range(len(allImages)):
				pixel = allImages[k][i][j][channel]
				weight = pixel / 64 if pixel < 64 else (255-pixel)/64 if pixel > 191 else 1

				sum += weight * (g_Z[pixel] - ln_ts[k])
				weight_sum += weight

			energy[i][j] = (sum / weight_sum) if weight_sum != 0 else 0

	return energy

def hdr(jsonPath, smooth=1, pixelNumber=None):

	with open(jsonPath) as f:
		imageInfos = json.load(f)

	allImages = []
	ln_ts = []

	# read all images and store as numpy.array
	for imageInfo in imageInfos:
		image = Image.open(path.join(path.dirname(jsonPath), imageInfo["path"]))
		allImages.append(np.array(image))
		ln_ts.append(math.log(imageInfo["t"], math.e))

	if not pixelNumber:
		pixelNumber = round(512 / len(allImages) * 1.1)

	print(f"Sample pixel: {pixelNumber}")

	outputs = []

	# RGB
	for channel in range(3):
		# Ax = b

		# A:
		# row: number of sample pixel * number of images, offset, smooth
		# column: g, En
		A = np.zeros((pixelNumber * len(allImages) +
					  1 + 254, 256 + pixelNumber))
		
		# b:
		# row: number of sample pixel * number of images, offset, smooth
		# column: 1
		b = np.zeros((A.shape[0]))

		# fill matrix
		for i in range(pixelNumber):
			pos = 0

			ok = False
			while not ok:
				pos = randint(0, allImages[0].shape[:2])
				ok=True
				for j in range(len(allImages)):
					if (allImages[j][pos[0]][pos[1]] == (0,0,255)).all():
						ok=False
						break

			for j in range(len(allImages)):
				pixel = allImages[j][pos[0]][pos[1]][channel]

				weight = pixel / 64 if pixel < 64 else (255 - pixel) / 64 if pixel > 191 else 1

				index = i * len(allImages) + j
				A[index][pixel] = weight
				A[index][256+i] = -weight
				b[index] = weight * ln_ts[j]

		A[pixelNumber * len(allImages)][127] = 1

		for i in range(254):
			index = pixelNumber * len(allImages) + 1 + i

			weight = pixel / 64 if pixel < 64 else (255-pixel)/64 if pixel > 191 else 1

			weight *= smooth ** 0.5

			A[index][i] = weight
			A[index][i+1] = -2 * weight
			A[index][i+2] = weight

		# x
		# row: g, En
		# column: 1
		x = lstsq(A, b, rcond=None)[0]

		g_Z = []
		for i in range(256):
			g_Z.append(x[i])


		energy = getLnEnergy(allImages, g_Z, ln_ts, channel)

		outputs.append(energy)

		plt.plot(g_Z, 'r' if channel==0 else 'g' if channel==1 else 'b')

	maxVal = max([np.amax(output) for output in outputs])
	minVal = min([np.amin(output) for output in outputs])
	outputs = [(output - minVal) * 255 / (maxVal - minVal) for output in outputs]

	output_image = np.zeros((outputs[0].shape[0], outputs[0].shape[1], 3), 'uint8')
	output_image[..., 0] = outputs[0]
	output_image[..., 1] = outputs[1]
	output_image[..., 2] = outputs[2]
	image = Image.fromarray(output_image)
	image.save("temp.png")
	image.show()
	plt.show()

	return outputs


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-j", "--jsonPath", type=str,
						help="The json file that store information about images", default="")
	parser.add_argument("-s", "--smooth", type=float,
						help="The weight of smooth parameter", default=1)
	parser.add_argument("-p", "--pixel", type=int,
						help="The number of sample for each g(x)", default=None)
	args = parser.parse_args()

	result = hdr(args.jsonPath, args.smooth, args.pixel)
	print(result)
