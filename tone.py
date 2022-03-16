from sys import maxsize
import numpy as np
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

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-j", "--jsonPath", type=str,
						help="The json file that store information about images", default="")
	parser.add_argument("-a", "--alpha", type=float,
						help="alpha", default=0.5)
	parser.add_argument("-l", "--local", action='store_true', 
						help="Using local operator")
	args = parser.parse_args()

	start_time = time.time()

	allImages, ln_ts = readJson(args.jsonPath)
	energys, g_Zs = hdr(allImages, ln_ts)
	luminances = globalOperator(energys, args.alpha, np.amax(energys) * 1.01)
	luminances = np.clip(luminances, 0, 1)
	
	maxVal = np.amax(luminances)
	minVal = np.amin(luminances)
	print("Global operator:")
	print(f"\tminVal: {minVal}")
	print(f"\tmaxVal: {maxVal}")

	image = Image.fromarray(np.around(luminances * 255).astype(np.uint8))
	image.show()

	if args.local:
		luminances, vs = localOperator(luminances)
		Image.fromarray(np.around(vs * 255).astype(np.uint8)).show()
		image = Image.fromarray(np.around(luminances * 255).astype(np.uint8))
		image.show()
		# display
		maxVal = np.amax(luminances)
		minVal = np.amin(luminances)
		print("Local operator:")
		print(f"\tminVal: {minVal}")
		print(f"\tmaxVal: {maxVal}")

	print(f"Spend {time.time() - start_time} sec")


	luminances = np.clip(luminances, 0, 1)
	image = Image.fromarray(np.around(luminances * 255).astype(np.uint8))
	image.save("temp.png")