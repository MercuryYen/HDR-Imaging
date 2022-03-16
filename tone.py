from sys import maxsize
import numpy as np
from torch import miopen_depthwise_convolution
from hdr import hdr
from alignment import toGrey
from readImage import readJson

from scipy.ndimage import gaussian_filter

import argparse
import time
from PIL import Image

from numba import njit

def globalOperator(energys, alpha = 0.18, Lwhite = 1.0):
	delta = 1e-6

	greyImage = toGrey(energys)
	Lw = np.exp(np.sum(np.log(greyImage + delta)) / greyImage.size)
	
	print(f"Lw: {Lw}")

	Lm = alpha * greyImage / Lw

	Ld = (Lm * (1 + Lm / Lwhite ** 2)) / (1 + Lm)

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

def localOperator(energys, alpha = 0.18, phi = 10, epsilon = 1, maxDepth = 30):
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

def localOperatorEx(energys, alpha = 0.18, phi = 10, target = 0.5, epsilon = 1, maxDepth = 30):
	gaussians = []
	for i in range(maxDepth):
		gaussians.append(gaussian_filter(toGrey(energys), sigma=i))
	gaussians = np.array(gaussians)
	
	maxEpsilon = 0.01
	minEpsilon = 0.01

	target = max(0.01, min(0.99, target))
	
	# find max epsilon
	temp, vs = getLd(np.copy(energys), np.zeros(gaussians[0].shape), gaussians, alpha, phi, maxEpsilon)
	while np.mean(vs) < target:
		maxEpsilon *= 2
		temp, vs = getLd(np.copy(energys), np.zeros(gaussians[0].shape), gaussians, alpha, phi, maxEpsilon)
		print(np.mean(vs), maxEpsilon)

	# find min epsilon
	temp, vs = getLd(np.copy(energys), np.zeros(gaussians[0].shape), gaussians, alpha, phi, minEpsilon)
	while np.mean(vs) > target:
		minEpsilon /= 2
		temp, vs = getLd(np.copy(energys), np.zeros(gaussians[0].shape), gaussians, alpha, phi, minEpsilon)
		print(np.mean(vs), minEpsilon)
	
	print(minEpsilon)
	print(maxEpsilon)

	# find final epsilon
	midEpsilon = (minEpsilon + maxEpsilon) / 2
	while maxEpsilon - minEpsilon > epsilon:
		midEpsilon = (minEpsilon + maxEpsilon) / 2
		temp, vs = getLd(np.copy(energys), np.zeros(gaussians[0].shape), gaussians, alpha, phi, maxEpsilon)
		
		if np.mean(vs)>target:
			maxEpsilon = midEpsilon
		else:
			minEpsilon = midEpsilon
		print(f"mean: {np.mean(vs)}")
		print(midEpsilon)
	energys, vs = getLd(np.copy(energys), np.zeros(gaussians[0].shape), gaussians, alpha, phi, maxEpsilon)
	
	return energys, vs
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-j", "--jsonPath", type=str,
						help="The json file that store information about images", default="")
	parser.add_argument("-s", "--smooth", type=float,
						help="The weight of smooth parameter", default=100)
	parser.add_argument("-p", "--pixel", type=int,
						help="The number of sample for each g(x)", default=500)
	parser.add_argument("-a", "--alpha", type=float,
						help="alpha", default=0.5)
	args = parser.parse_args()

	start_time = time.time()

	allImages, ln_ts = readJson(args.jsonPath)
	energys, g_Zs = hdr(allImages, ln_ts, args.smooth, args.pixel)
	luminances = globalOperator(energys, args.alpha, np.amax(energys) * 1.01)
	luminances = np.clip(luminances, 0, 1)
	
	maxVal = np.amax(luminances)
	minVal = np.amin(luminances)
	print("Global operator:")
	print(f"\tminVal: {minVal}")
	print(f"\tmaxVal: {maxVal}")

	Image.fromarray(np.around(luminances * 255).astype(np.uint8)).show()

	luminances, vs = localOperator(luminances, alpha = 1.6, phi = 8, epsilon = 1e-4, maxDepth = 10)
	# luminances, vs = localOperatorEx(luminances, alpha = 1.6, phi = 8, target = 0.6, epsilon = 1e-6, maxDepth = 60)
	# outputs = [energy*luminance for energy, luminance in zip(energys,luminances)]

	print(f"Spend {time.time() - start_time} sec")

	# display
	maxVal = np.amax(luminances)
	minVal = np.amin(luminances)
	print("Local operator:")
	print(f"\tminVal: {minVal}")
	print(f"\tmaxVal: {maxVal}")

	luminances = np.clip(luminances, 0, 1)
	Image.fromarray(np.around(vs * 255).astype(np.uint8)).show()
	image = Image.fromarray(np.around(luminances * 255).astype(np.uint8))
	image.show()
	image.save("temp.png")