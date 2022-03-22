import numpy as np
from numpy.linalg import lstsq
from numpy.random import randint

from PIL import Image

import matplotlib.pyplot as plt

from numba import njit

import time

from readImage import readJson

import argparse

def writeHDR(image, filename):
	f = open(filename, "wb")
	f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
	f.write("-Y {0} +X {1}\n".format(image.shape[0], image.shape[1]).encode())

	brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
	mantissa = np.zeros_like(brightest)
	exponent = np.zeros_like(brightest)
	np.frexp(brightest, mantissa, exponent)
	scaled_mantissa = mantissa * 256.0 / brightest
	rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
	rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
	rgbe[...,3] = np.around(exponent + 128)

	rgbe.flatten().tofile(f)
	f.close()

@njit
def buildAb(allImages, ln_ts, smooth, channel, pixels, A: np.array, b: np.array):
	for i in range(len(pixels)):
		pos = pixels[i]

		for j in range(len(allImages)):
			pixel = allImages[j][pos[0]][pos[1]][channel]

			weight = pixel / \
				64 if pixel < 64 else (255 - pixel) / 64 if pixel > 191 else 1

			index = i * len(allImages) + j
			A[index][pixel] = weight
			A[index][256+i] = -weight
			b[index] = weight * ln_ts[j]

	A[len(pixels) * len(allImages)][127] = 1

	for i in range(254):
		index = len(pixels) * len(allImages) + 1 + i

		weight = pixel / \
			64 if pixel < 64 else (255-pixel)/64 if pixel > 191 else 1

		weight *= smooth ** 0.5

		A[index][i] = weight
		A[index][i+1] = -2 * weight
		A[index][i+2] = weight

	return A, b


@njit
def getEnergy(allImages, g_Z, ln_ts, channel):
	energy = np.zeros(allImages[0].shape[:2], dtype='float64')
	for i in range(energy.shape[0]): 							# x
		for j in range(energy.shape[1]): 						# y

			sum = 0
			weight_sum = 0
			for k in range(len(allImages)):						# picture index
				pixel = allImages[k][i][j][channel]
				weight = pixel / \
					64 if pixel < 64 else (255-pixel)/64 if pixel > 191 else 1

				sum += weight * (g_Z[pixel] - ln_ts[k])
				weight_sum += weight

			if weight_sum == 0:
				sum = (min(g_Z) - max(ln_ts) if pixel == 0 else max(g_Z) - min(ln_ts))
				weight_sum = 1

			energy[i][j] = np.exp(sum / weight_sum)

	return energy


# imagesInfo:[
# 	{
#		"path": "path/to/image",
#		"t": 0.3
#	}
# ]
def hdr(allImages, ln_ts, smooth=10, pixelNumber=None, ghost_removal = True):

	if not pixelNumber:
		pixelNumber = round(8192 / len(allImages) * 1.1)

	print(f"Sample pixel: {pixelNumber}")

	outputs = []
	g_Zs = []

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

		# generate random pixel
		pixels = []
		for i in range(pixelNumber):
			pos = randint(0, allImages[0].shape[:2])
			pixels.append(pos)
		pixels = np.array(pixels)

		# fill matrix
		A, b = buildAb(allImages, ln_ts, smooth, channel, pixels, A, b)

		# x
		# row: g, En
		# column: 1
		x = lstsq(A, b, rcond=None)[0]

		g_Z = []
		for i in range(256):
			g_Z.append(x[i])
		g_Z = np.array(g_Z)

		energy = getEnergy(allImages, g_Z, ln_ts, channel)

		outputs.append(energy)
		g_Zs.append(g_Z)

	outputs = np.array(outputs).transpose([1, 2, 0])

	if ghost_removal:
		

	return outputs, g_Zs

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-j", "--jsonPath", type=str,
						help="The json file that store information about images", default="")
	parser.add_argument("-s", "--smooth", type=float,
						help="The weight of smooth parameter", default=1)
	parser.add_argument("-p", "--pixel", type=int,
						help="The number of sample for each g(x)", default=None)
	args = parser.parse_args()

	start_time = time.time()

	allImages, ln_ts = readJson(args.jsonPath)
	outputs, g_Zs = hdr(allImages, ln_ts, args.smooth, args.pixel)

	print(f"Spend {time.time() - start_time} sec")

	writeHDR(outputs, "temp.hdr")

	outputs = np.maximum(outputs, 0.001)

	# display
	outputs = np.log(outputs)
	maxVal = np.amax(outputs)
	minVal = np.amin(outputs)

	print(np.amin(outputs))
	print(np.amax(outputs))

	outputs = outputs / maxVal
	outputs = np.clip(outputs, 0, 1)
	outputs = outputs * 255
	
	image = Image.fromarray(np.around(outputs).astype(np.uint8))
	image.save("temp.png")
	image.show()

	plt.plot(g_Zs[0], "r")
	plt.plot(g_Zs[1], "g")
	plt.plot(g_Zs[2], "b")
	plt.show()
