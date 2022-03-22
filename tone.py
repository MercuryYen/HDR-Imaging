import numpy as np
from hdr import hdr
from alignment import toGrey
from readImage import readJson
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt

import argparse
import time
from PIL import Image

from numba import njit

def logMap(energys, b = 1.3):
	delta = 1e-6
	Lw = toGrey(energys)
	Lave = np.exp(np.sum(np.log(Lw + delta)) / Lw.size)
	print(Lave)


	LwMax = np.max(Lw) / Lave * 1.1

	Ld = np.log(Lw + 1) / np.log10(LwMax + 1) / np.log(2 + (Lw / LwMax) ** (np.log(b) / np.log(0.5)) * 8)
	print(np.min(Ld))
	print(np.max(Ld))
	
	output = np.copy(energys)
	for i in range(output.shape[2]):
		output[:, :, i] = output[:, :, i] / Lw * Ld

	return output

def globalOperator(energys, alpha = 0.18, Lwhite = 1.0):
	delta = 1e-6

	greyImage = toGrey(energys)
	Lw = np.exp(np.sum(np.log(greyImage + delta)) / greyImage.size)
	
	print(f"Lw: {Lw}")

	Lm = alpha * greyImage / Lw

	#Ld = Lm / (1 + Lm)
	Ld = (Lm * (1 + Lm / Lwhite ** 2)) / (1 + Lm)
	print(Ld.shape)

	output = np.copy(energys)
	for i in range(output.shape[2]):
		output[:, :, i] = output[:, :, i] / greyImage * Ld

	return output

@njit
def getLd(greyImage, vs, gaussians, alpha, phi, epsilon):
	for y in range(greyImage.shape[0]):
		for x in range(greyImage.shape[1]):
			maxS = 0
			for s in range(1, len(gaussians)):
				v = (gaussians[s - 1][y][x] - gaussians[s][y][x])/(alpha * (2**phi) / (s * s) + gaussians[s-1][y][x])
				if np.abs(v) < epsilon:
					maxS = s
				else:
					break
			vs[y][x] = (maxS - 1) / (len(gaussians)-2)
			greyImage[y][x] = greyImage[y][x] / (1 + gaussians[maxS - 1][y][x])
	return greyImage, vs

def localOperator(energys, alpha = 1.6, phi = 8, epsilon = 1e-4, maxDepth = 30):
	gaussians = []
	greyImage = toGrey(energys)
	for i in range(maxDepth):
		gaussians.append(gaussian_filter(greyImage, sigma=i))
	gaussians = np.array(gaussians)
	Ld, vs = getLd(np.copy(greyImage), np.zeros(gaussians[0].shape), gaussians, alpha, phi, epsilon)
	
	output = np.copy(energys)
	for i in range(output.shape[2]):
		output[:, :, i] = output[:, :, i] / greyImage * Ld

	return output, vs

def getGaussuanFilter(size, sigma):
	g = np.zeros((size, size))
	sigma2 = sigma ** 2
	halfSize = int(size / 2)
	sum = 0
	for i in range(-halfSize, halfSize + 1):
		for j in range(-halfSize, halfSize + 1):
			g[i + halfSize][j + halfSize] = np.exp(-(i ** 2 + j ** 2) / (2 * sigma2)) / (2 * np.pi * sigma2)
			sum += g[i + halfSize][j + halfSize]
	print(sum)
	return g

def multiFilter(I, g):
	for i in range(I.shape(0)):
		for j in range(I.shape(1)):
			sum = 0
			for x in range(g.shape(0)):
				for y in range(g.shape(1)):
					u = x + i
					v = y + j
					if (u < 0 or u >= I.shape(0) or v < 0 or v >= I.shape(1)):
						continue
					sum += I[u][v] * g[x][y]

def fastBilateralFiltering(energys):
	delta = 0.000001

	I = np.log(delta + toGrey(energys))
	g = getGaussuanFilter(3,1)
	print(g)
	L = gaussian_filter(I, sigma=1)
	Rs= np.copy(energys)
	# RGB
	for c in range(3):
		Rs[:, :, c] = energys[:, :, c] / L
	outputs = Rs
	return outputs

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-j", "--jsonPath", type=str,
						help="The json file that store information about images", default="")
	parser.add_argument("-a", "--alpha", type=float,
						help="alpha", default=0.18)
	parser.add_argument("-l", "--local", action='store_true', 
						help="Using local operator")
	parser.add_argument("-f", "--fbf", action='store_true', 
						help="Using fast Bilateral filtering")
	parser.add_argument("-lm", "--logMap", action='store_true', 
						help="Using log map")
	parser.add_argument("-b", "--bias", type=float,
						help="bias", default=1.2)
	args = parser.parse_args()

	start_time = time.time()

	allImages, ln_ts = readJson(args.jsonPath)
	energys, g_Zs = hdr(allImages, ln_ts)
	# image.show()
	# for c in range(3):
	# 	Image.fromarray(np.around(luminances[:, :, c] * 255).astype(np.uint8)).show()

	if args.fbf:
		print("Using Fast Bilateral filtering")
		outputs = fastBilateralFiltering(energys)
		image = Image.fromarray(np.around(outputs).astype(np.uint8))
		image.show()

	elif args.logMap:
		print("Using log map")
		outputs = logMap(energys, args.bias)
		maxVal = np.amax(outputs)
		minVal = np.amin(outputs)
		print(f"\tminVal: {minVal}")
		print(f"\tmaxVal: {maxVal}")
		outputs = np.clip(outputs, 0, 1)
		image = Image.fromarray(np.around(outputs * 255).astype(np.uint8))
		image.show()

	else:
		print("Using Photographic Tone")
		print(np.amax(energys))
		luminances = globalOperator(energys, args.alpha, np.amax(energys))
		
		maxVal = np.amax(luminances)
		minVal = np.amin(luminances)
		print("Global operator:")
		print(f"\tminVal: {minVal}")
		print(f"\tmaxVal: {maxVal}")

		luminances = np.clip(luminances, 0, 1)

		image = Image.fromarray(np.around(luminances * 255).astype(np.uint8))
		image.show()

		if args.local:
			print("with Dodging-and-burning")
			luminances, vs = localOperator(luminances)
			Image.fromarray(np.around(vs * 255).astype(np.uint8)).show()

			# display
			print("Local operator:")
			print(f"\tminVal: {minVal}")
			print(f"\tmaxVal: {maxVal}")
			luminances = np.clip(luminances, 0, 1)
			image = Image.fromarray(np.around(luminances * 255).astype(np.uint8))
			image.show()

	print(f"Spend {time.time() - start_time} sec")
	
	image.save("temp.png")