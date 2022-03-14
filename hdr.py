from cv2 import exp
import numpy as np
from numpy.linalg import lstsq
from numpy.random import randint

from PIL import Image

import matplotlib.pyplot as plt

from numba import njit

import time

from readImage import readJson

import argparse


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

			energy[i][j] = np.exp(sum / weight_sum) if weight_sum != 0 else 1e-4

	return energy


# imagesInfo:[
# 	{
#		"path": "path/to/image",
#		"t": 0.3
#	}
# ]

def hdr(allImages, ln_ts, smooth=1, pixelNumber=None):

	if not pixelNumber:
		pixelNumber = round(512 / len(allImages) * 1.1)

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

	return outputs, g_Zs

def globalOperator(energys, alpha = 0.8, Lwhite = 1.0):
	outputs = []
	delta = 0.000001
	for c in range(len(energys)): 						# chennel
		Lw = 0
		for i in range(energys[c].shape[0]): 					# y
			for j in range(energys[c].shape[1]): 				# x
				Lw += np.log(delta + energys[c][i][j])
		Lw /= i * j
		Lw = np.exp(Lw)
		Lms = []
		Lds = np.zeros(energys[c].shape[:2], dtype='float64')
		for i in range(energys[c].shape[0]): 					# y
			for j in range(energys[c].shape[1]): 				# x
				Lm = alpha * energys[c][i][j] / Lw
				Lms.append(Lm)
				Ld = (Lm * (1 + Lm / Lwhite ** 2)) / (1 + Lm)
				# Ld = Lm / (1 + Lm)
				Lds[i][j] = Ld
		outputs.append(Lds)
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

	start_time = time.time()

	allImages, ln_ts = readJson(args.jsonPath)
	energys, g_Zs = hdr(allImages, ln_ts, args.smooth, args.pixel)
	luminances = globalOperator(energys, 0.18, 255)
	# outputs = [energy*luminance for energy, luminance in zip(energys,luminances)]
	outputs = luminances

	print(f"Spend {time.time() - start_time} sec")

	# display
	maxVal = max([np.amax(np.log(output)) for output in outputs])
	minVal = min([np.amin(np.log(output)) for output in outputs])
	print(min([np.amin(output) for output in outputs]))
	print(max([np.amax(output) for output in outputs]))
	outputs = [(np.log(output) - minVal) * 255 / (maxVal - minVal)
			   for output in outputs]

	output_image = np.zeros(
		(energys[0].shape[0], energys[0].shape[1], 3), 'uint8')
	for i in range(3):
		output_image[..., i] = outputs[i]
	image = Image.fromarray(output_image)
	image.save("temp.png")
	image.show()

	plt.plot(g_Zs[0], "r")
	plt.plot(g_Zs[1], "g")
	plt.plot(g_Zs[2], "b")
	plt.show()
