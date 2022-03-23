import numpy as np
from hdr import hdr
from alignment import alignment, toGrey, halfBitmap
from readImage import readJson
from scipy.ndimage import gaussian_filter

import argparse
import time
from PIL import Image

from numba import njit

def rgb_Yxy(energys):
	table = np.transpose(np.array([[0.5141364, 0.3238786, 0.16036376],
                         [0.265068, 0.67023428, 0.06409157],
                         [0.0241188, 0.1228178, 0.84442666]
						 ]))
	result = np.matmul(energys, table)

	W = np.sum(result, axis = 2)
	temp = W > 0
	output = np.copy(energys)
	output[:, :, 0] = temp * result[:, :, 1]
	output[:, :, 1] = temp * result[:, :, 0] / W
	output[:, :, 2] = temp * result[:, :, 1] / W

	return output

def Yxy_rgb(energys):
	table = np.transpose(np.array([[2.5651, -1.1665, -0.3986],
						 [-1.0217, 1.9777, 0.0439], 
						 [0.0753, -0.2543, 1.1892]
						]))
	Y = energys[:, :, 0]
	result1 = energys[:, :, 1]
	result2 = energys[:, :, 2]
	temp = ((Y > 1e-6) & (result1 > 1e-6) & (result2 > 1e-6))
	X = temp * result1 * Y / result2
	Z = temp * X / result1 - X - Y
	output = np.copy(energys)
	output[:, :, 0] = X
	output[:, :, 1] = Y
	output[:, :, 2] = Z

	return np.matmul(output, table)


def logMap(energys, b = 1.3):
	output = np.copy(energys)

	delta = 1e-6
	output = rgb_Yxy(output)
	Lw = output[:, :, 0]
	Lave = np.exp(np.sum(np.log(Lw + delta)) / Lw.size)

	LwMax = np.max(Lw) / Lave

	Lw = Lw / Lave
	Lw = np.log(Lw + 1) / np.log10(LwMax + 1) / np.log(2 + (Lw / LwMax) ** (np.log(b) / np.log(0.5)) * 8)
	print(np.min(Lw))
	print(np.max(Lw))
	
	output[:, :, 0] = Lw
	output = Yxy_rgb(output)
	#output = np.copy(energys)
	#for i in range(output.shape[2]):
	#	output[:, :, i] = output[:, :, i] / Lw * Ld

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

@njit
def getGaussuan(x2, y2, sigma2):
	return np.exp(-(x2 + y2) / (2 * sigma2)) / (2 * np.pi * sigma2)

@njit
def getGaussuanFilter(sizeX, sizeY, sigma):
	g = np.zeros((sizeX, sizeY))
	sigma2 = sigma ** 2
	halfSizeX = int(sizeX / 2)
	halfSizeY = int(sizeY / 2)
	sum = 0
	for i in range(sizeX):
		for j in range(sizeY):
			g[i][j] = getGaussuan(abs(i - halfSizeX)**2, abs(j - halfSizeY)**2, sigma2)
	sum = np.sum(g)
	# print(g)
	g = g / sum
	return g

@njit
def getGaussuanFilterI(sizeX, sizeY, sigma, I, centerX, centerY):
	g = np.zeros((sizeX, sizeY))
	sigma2 = sigma ** 2
	halfSizeX = int(sizeX / 2)
	halfSizeY = int(sizeY / 2)
	sum = 0
	for i in range(sizeX):
		for j in range(sizeY):
			x = centerX + i - halfSizeX
			y = centerY + j - halfSizeY
			if (x < 0):
				x = -x
			elif (x >= I.shape[0]):
				x = I.shape[0] - (x - I.shape[0]) - 1
			elif (y < 0):
				y = -y
			elif (y >= I.shape[1]):
				y = I.shape[1] - (y - I.shape[1]) - 1
			g[i][j] = getGaussuan(I[centerX][centerY]**2, I[x][y]**2, sigma2)
	sum = np.sum(g)
	# print(g)
	g = g / sum
	return g

@njit
def multiFilter(I, g):
	output = I
	for i in range(I.shape[0]):
		for j in range(I.shape[1]):
			sum = 0
			for x in range(g.shape[0]):
				for y in range(g.shape[1]):
					u = x + i
					v = y + j
					if (u < 0 or u >= I.shape[0] or v < 0 or v >= I.shape[1]):
						continue
					sum += I[u][v] * g[x][y]
			output[i][j] = sum
	return output

@njit
def bilateralFiltering(Itensity, f:np.broadcast_arrays, kernal_half_size = 2, sigma_g = 0.4):
	rows, cols = Itensity.shape
	logFiltered = np.zeros(Itensity.shape, dtype='float64')
	for r in range(rows):
		for c in range(cols):
			ks_sum = 0
			js_sum = 0
			for ii in range(-kernal_half_size, kernal_half_size + 1):
				if ii + c >= 0 and ii + c < cols:
					for jj in range(-kernal_half_size, kernal_half_size + 1):
						if jj + r >= 0 and jj + r < rows:
							# compute weight here
							i_p = Itensity[r][c]
							i_s = Itensity[r+jj][c+ii]
							weight = f[jj + kernal_half_size][ii + kernal_half_size] * getGaussuan((i_p - i_s) * (i_p - i_s), 0, sigma_g ** 2)
							ks_sum += weight
							js_sum += weight * i_s

			logFiltered[r][c] = js_sum / ks_sum

	return logFiltered

def fastBilateralFiltering(energys, sigma_f = 2, sigma_g = 0.4, compressFactor = 0.25):
	energys = energys / np.max(energys) * 255
	I = toGrey(energys)
	logI = np.log10(I)
	Colors= np.copy(energys)
	# RGB
	for c in range(3):
		Colors[:, :, c] = energys[:, :, c] / I

	# print(np.max(Colors))	
	# print(np.min(Colors))

	x = np.zeros((sigma_f * 2 + 1, sigma_f * 2 + 1), dtype=np.float64)
	x[sigma_f, sigma_f] = 1
	f = gaussian_filter(x, sigma=sigma_f)

	logBase = bilateralFiltering(logI, f, sigma_f, sigma_g)
	logDetail = logI - logBase
	logNewI = logBase * compressFactor + logDetail
	newI = np.power(10, logNewI)

	outputs = np.copy(Colors)
	
	# RGB
	for c in range(3):
		outputs[:, :, c] = Colors[:, :, c] * newI[:, :]
	
	outputs = np.clip(outputs, 0, 1)
	outputs = outputs * 255
	
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
						help="bias", default=0.65)
	parser.add_argument("-c", "--compress", type=float,
						help="compress", default=0.25)
	parser.add_argument("-d", "--shiftDepth", type=int,
						help="The depth of shift recursive.", default=3)
	parser.add_argument("-i", "--iteration", type=int,
						help="Half image iteration", default=0)
	parser.add_argument("-k", "--kernalSize", type=int,
						help="kernal size", default=2)
	parser.add_argument("-n", "--fileName", type=str,
						help="output HDR file name", default='output')
	parser.add_argument("-ngr", "--notGhostRemoval", action='store_false', 
						help="Not Using Ghost removal")
	args = parser.parse_args()

	start_time = time.time()

	allImages, ln_ts = readJson(args.jsonPath)

	if args.iteration > 0:
		newAllImages = []
		for i in range(len(allImages)):
			target = allImages[i]
			for iter in range(args.iteration):
				target = halfBitmap(target)
			newAllImages.append(target)
		allImages = np.array(newAllImages)

	allImages = alignment(allImages, args.shiftDepth)

	energys, g_Zs = hdr(allImages, ln_ts, ghostRemoval=args.notGhostRemoval)
	# image.show()
	# for c in range(3):
	# 	Image.fromarray(np.around(luminances[:, :, c] * 255).astype(np.uint8)).show()

	if args.fbf:
		print('Using Bilateral Tone')
		outputs = fastBilateralFiltering(energys, args.kernalSize, 0.4, args.compress)

		minVal = np.amin(outputs)
		maxVal = np.amax(outputs)
		print(f"\tminVal: {minVal}")
		print(f"\tmaxVal: {maxVal}")

		image = Image.fromarray(np.around(outputs).astype(np.uint8))
		image.show()

	elif args.logMap:
		print("Using log map")
		outputs = logMap(energys, args.bias)
		minVal = np.amin(outputs)
		maxVal = np.amax(outputs)
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