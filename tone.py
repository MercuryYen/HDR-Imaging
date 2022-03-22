import numpy as np
from hdr import hdr
from alignment import toGrey
from readImage import readJson
from scipy.ndimage import gaussian_filter
from scipy import ndimage
from scipy import signal

import matplotlib.pyplot as plt

import argparse
import time
import math
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

def bilateralFiltering(Itensity):
	sigma_f = 2
	sigma_g = 0.4
	# dx = ndimage.sobel(Itensity, 0)  # horizontal derivative
	# dy = ndimage.sobel(Itensity, 1)  # vertical derivative
	# i = np.hypot(dx, dy)  # magnitude
	x = np.zeros((5, 5), dtype=np.float64)
	x[2, 2] = 1
	f = gaussian_filter(x, sigma=sigma_f)
	rows, cols = Itensity.shape
	logFiltered = np.zeros(Itensity.shape, dtype='float64')
	filtered = np.zeros(Itensity.shape, dtype='float64')
	for r in range(rows):
		for c in range(cols):
			ks = []
			js = []
			# for ii in range(-math.ceil(2*sigma_f), math.ceil(2*sigma_f)+1):
			for ii in range(-2, 3):
				if ii + c >= 0 and ii + c < cols:
					# for jj in range(-math.ceil(2*sigma_f), math.ceil(2*sigma_f)+1):
					for jj in range(-2, 3):
						if jj + r >= 0 and jj + r < rows:
							# compute weight here
							i_p = Itensity[r][c]
							i_s = Itensity[r+jj][c+ii]
							# weight = gaussian_filter(math.sqrt(ii**2 + jj**2), sigma_f) * gaussian_filter(i_p - i_s, sigma_g)
							weight = f[jj + 2][ii + 2] * getGaussuan(i_p ** 2, i_s ** 2, sigma_g ** 2)
							ks.append(weight)
							js.append(weight * i_p)
			ks_sum = np.sum(np.asarray(ks))
			js_sum = np.sum(np.asarray(js))
			logFiltered[r][c] = js_sum / ks_sum
			# filtered[r][c] = js_sum / ks_sum
			filtered[r][c] = math.pow(10, logFiltered[r][c])

	return logFiltered, filtered

def fastBilateralFiltering(energys):
	I = toGrey(energys)
	logI = np.log10(I)
	Colors= np.copy(energys)
	# RGB
	for c in range(3):
		Colors[:, :, c] = energys[:, :, c] / I

	logBase, base = bilateralFiltering(logI)
	logDetail = logI - logBase
	outputs = np.copy(Colors)
	
	# RGB
	for c in range(3):
		outputs[:, :, c] = Colors[:, :, c] * base[:, :]

	# outputs = np.clip(outputs, 0, 1)
	# outputs = outputs * 255
	# ========================need fix b============================
	# hdrImage = np.clip(energys, 0, 1)
	# hdrImage = hdrImage * 255
		
	# I = toGrey(hdrImage)
	# Colors= np.copy(energys)
	# # RGB
	# for c in range(3):
	# 	Colors[:, :, c] = hdrImage[:, :, c] / I
	# # Colors = np.clip(Colors, 0, 1)
	# # Colors = Colors * 255

	# logBase, base = bilateralFiltering(I)
	# # outputs = Colors
	# detail = base / I
	# outputs = np.copy(Colors)
	
	# # RGB
	# for c in range(3):
	# 	# outputs[:, :, c] = Colors[:, :, c] * base[:, :]
	# 	outputs[:, :, c] = base[:, :]
	# ========================error============================
	# I = toGrey(energys)
	# logI = np.log10(I)
	# Colors= np.copy(energys)
	# # RGB
	# for c in range(3):
	# 	Colors[:, :, c] = np.log10(energys[:, :, c]) - logI
	# # Colors = np.clip(Colors, 0, 1)
	# # Colors = Colors * 255

	# # logBase, base = bilateralFiltering(logI)s
	# # logBase, base = bilateralFiltering(toGrey(hdrImage))
	# # logBase, base = bilateralFiltering(I)
	# logBase = gaussian_filter(logI, sigma=1)
	# # outputs = Colors
	# logDetail = logBase - logI
	# outputs = np.copy(Colors)
	
	# # RGB
	# for c in range(3):
	# 	outputs[:, :, c] = Colors[:, :, c] + logBase[:, :] + logDetail[:, :]

	# outputs = np.clip(outputs, 0, 1)
	# outputs = outputs * 255
	# ==========================gaussian==============================
	# I = toGrey(energys)
	# logI = np.log10(I)
	# Colors= np.copy(energys)
	# # RGB
	# for c in range(3):
	# 	Colors[:, :, c] = energys[:, :, c] / I
	# # Colors = np.clip(Colors, 0, 1)
	# # Colors = Colors * 255

	# # base = gaussian_filter(I, sigma=1)
	# # g = getGaussuanFilter(energys.shape[0], energys.shape[1], 1)
	# g = getGaussuanFilter(5, 5, 1)
	# # base = multiFilter(I, g)
	# base = signal.convolve2d(I, g, boundary='symm', mode='same')
	# # outputs = Colors
	# detail = base / I
	# outputs = np.copy(Colors)
	
	# # RGB
	# for c in range(3):
	# 	outputs[:, :, c] = Colors[:, :, c] * base[:, :]

	# outputs = np.clip(outputs, 0, 1)
	# outputs = outputs * 255
	# =============================================================
	# hdrImage = np.clip(energys, 0, 1)
	# hdrImage = hdrImage * 255
		
	# I = toGrey(hdrImage)
	# logI = np.log10(I)
	# Colors= np.copy(energys)
	# # RGB
	# for c in range(3):
	# 	Colors[:, :, c] = hdrImage[:, :, c] / I
	# # Colors = np.clip(Colors, 0, 1)
	# # Colors = Colors * 255

	# # logBase, base = bilateralFiltering(logI)s
	# # logBase, base = bilateralFiltering(toGrey(hdrImage))
	# # logBase, base = bilateralFiltering(I)
	# base = gaussian_filter(I, sigma=1)
	# # outputs = Colors
	# detail = base / I
	# outputs = np.copy(Colors)
	
	# # RGB
	# for c in range(3):
	# 	outputs[:, :, c] = Colors[:, :, c] * base[:, :] * detail[:, :]
	# ================================================================


	

	# logDetail = logI - logBase
	# maxLogBase = np.max(logBase)
	# minLogBase = np.min(logBase)
	# compressFactor = np.log10(50) / (maxLogBase - minLogBase)
	# scaleFactor = maxLogBase * compressFactor
	# logIntensity = logBase * compressFactor - scaleFactor + logDetail
	# intensity = np.copy(logIntensity)
	
	# for r in range(logIntensity.shape[0]):
	# 	for c in range(logIntensity.shape[1]):
	# 		intensity[r][c] = math.pow(10, logIntensity[r][c])
	# outputs = np.copy(energys)
	# # RGB
	# for c in range(3):
	# 	outputs[:, :, c] = Colors[:, :, c] * intensity[:, :]



	# logI = np.log10(delta + I)
	# # I = toGrey(energys)
	# # g = getGaussuanFilter(3,1)
	# # print(g)
	# x = np.zeros((11, 11), dtype=np.float64)
	# x[5, 5] = 1
	# g = gaussian_filter(x, sigma=1)
	# result = ndimage.sobel(energys)
	# print('result')
	# print(result)
	# # L = multiFilter(I, g)
	# # logBase = signal.convolve2d(logI, g, boundary='symm', mode='same') #卷積
	# base = signal.convolve2d(I, g, boundary='symm', mode='same') #卷積
	# detail = I / base	# I = illuminance * reflectance
	# # L = gaussian_filter(I, sigma=1)
	

	# logDetail = logI - base

	# outputs= np.copy(energys) # initial
	# # RGB
	# for c in range(3):
	# 	# outputs[:, :, c] = detail[:, :]
	# 	colorDetail = energys[:,:,c] / base[:,:]	# I = illuminance * reflectance
	# 	# outputs[:, :, c] = Colors[:, :, c] * colorDetail[:,:]

	# 	# Colors[:, :, c] = logI[:, :]
	# outputs = colorDetail
	# print(np.amin(outputs))
	# print(np.amax(outputs))
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
		print('calculating Fast Bilateral Filtering...')
		outputs = fastBilateralFiltering(energys)

		# maxVal = np.amax(outputs)
		# minVal = np.amin(outputs)

		# print(np.amin(outputs))
		# print(np.amax(outputs))

		# # outputs = outputs / maxVal
		# outputs = np.clip(outputs, 0, 1)
		# outputs = outputs * 255

		image = Image.fromarray(np.around(outputs).astype(np.uint8))
		image.show()

		# outputs = np.clip(outputs, 0, 1)
		# outputs = outputs * 255

		# image = Image.fromarray(np.around(outputs).astype(np.uint8))
		# image.show()

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