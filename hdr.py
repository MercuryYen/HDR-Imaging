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

	brightest = np.maximum(np.maximum(
		image[..., 0], image[..., 1]), image[..., 2])
	mantissa = np.zeros_like(brightest)
	exponent = np.zeros_like(brightest)
	np.frexp(brightest, mantissa, exponent)
	scaled_mantissa = mantissa * 256.0 / brightest
	rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
	rgbe[..., 0:3] = np.around(image[..., 0:3] * scaled_mantissa[..., None])
	rgbe[..., 3] = np.around(exponent + 128)

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

		weight = i / \
			64 if i < 64 else (255-i)/64 if i > 191 else 1

		weight *= smooth ** 0.5

		A[index][i] = weight
		A[index][i+1] = -2 * weight
		A[index][i+2] = weight

	return A, b


@njit
def getEnergy(allImages, g_Z, ln_ts, channel, ghostRemoval = True):
	energy = np.zeros(allImages[0].shape[:2], dtype='float64')
	energyBox = np.zeros(allImages.shape[0], dtype='float64')
	weightBox = np.zeros(allImages.shape[0], dtype='float64')
	
	for i in range(energy.shape[0]): 							# x
		for j in range(energy.shape[1]): 						# y
			sum = 0
			weight_sum = 0
			
			for k in range(len(allImages)):
				energyBox[k] = g_Z[allImages[k][i][j][channel]] - ln_ts[k]
			for k in range(len(allImages)):						# picture index
				pixel = allImages[k][i][j][channel]
				weight = pixel / \
					64 if pixel < 64 else (255-pixel)/64 if pixel > 191 else 1
				weightBox[k] = weight
			
			sum = np.sum(energyBox * weightBox)
			weight_sum = np.sum(weightBox)
			
			if weight_sum == 0:
				sum = (min(g_Z) - max(ln_ts) if pixel ==
					0 else max(g_Z) - min(ln_ts))
				weight_sum = 1

			if ghostRemoval:
				mean =  sum / weight_sum
				weightBox /= np.clip(np.power(np.exp(energyBox) - np.exp(mean), 10), 1e-10, 1e10)

				sum = np.sum(energyBox * weightBox)
				weight_sum = np.sum(weightBox)

				if weight_sum == 0:
					sum = (min(g_Z) - max(ln_ts) if pixel ==
						0 else max(g_Z) - min(ln_ts))
					weight_sum = 1
			
			energy[i][j] = np.exp(sum / weight_sum)

	return energy


# imagesInfo:[
# 	{
#		"path": "path/to/image",
#		"t": 0.3
#	}
# ]
def hdr(allImages, ln_ts, smooth=10, pixelNumber=None, ghostRemoval=True):

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

		energy = getEnergy(allImages, g_Z, ln_ts, channel, ghostRemoval)

		outputs.append(energy)
		g_Zs.append(g_Z)

	outputs = np.array(outputs).transpose([1, 2, 0])

	return outputs, g_Zs

def selectRef(allImages, exposureThres):
	thresholdCount = []
	lowThres = exposureThres
	highThres = (1 - exposureThres)
	for i in range(allImages.shape[0]):
		thresholdCount.append(np.count_nonzero(
			(allImages[i] > lowThres) & (allImages[i] < highThres)))
	thresholdCount = np.array(thresholdCount)
	return np.argmax(thresholdCount)

def hist_norm(source, template):

	olddtype = source.dtype
	oldshape = source.shape
	source = source.ravel()
	template = template.ravel()

	s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
											return_counts=True)
	t_values, t_counts = np.unique(template, return_counts=True)
	s_quantiles = np.cumsum(s_counts).astype(np.float64)
	s_quantiles /= s_quantiles[-1]
	t_quantiles = np.cumsum(t_counts).astype(np.float64)
	t_quantiles /= t_quantiles[-1]
	interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
	interp_t_values = interp_t_values.astype(olddtype)

	return interp_t_values[bin_idx].reshape(oldshape)


@njit
def convolution2d(image, kernel):
	y, x = image.shape
	y = y - kernel.shape[0] + 1
	x = x - kernel.shape[1] + 1
	new_image = np.zeros((y, x))
	for i in range(y):
		for j in range(x):
			new_image[i][j] = np.sum(
				image[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
	return new_image


@njit
def convolution3d(image, kernel):
	z, y, x = image.shape
	z = z - kernel.shape[0] + 1
	y = y - kernel.shape[1] + 1
	x = x - kernel.shape[2] + 1
	new_image = np.zeros((z, y, x))
	for i in range(z):
		for j in range(y):
			for k in range(x):
				new_image[i][j][k] = np.sum(
					image[i:i+kernel.shape[0], j:j+kernel.shape[1], k:k+kernel.shape[2]] * kernel)
	return new_image

def imfConsistenct(mu, refIdx, consistenctThres):
	cMap = np.zeros(mu.shape)
	cMap[refIdx] = np.ones([mu.shape[1], mu.shape[2]])

	refMu = mu[refIdx]
	for i in range(mu.shape[0]):
		if i != refIdx:
			cMu = hist_norm(mu[i], refMu)
			diff = np.abs(cMu - refMu)
			cMap[i] = diff <= consistenctThres
	return cMap

def spdMef(allImages, p=4, gSig=0.2, lSig=0.5, wSize=21, stepSize=2, exposureThres=0.01, consistencyThres=0.1, structureThres=0.8, C=0.00045):
	allImages = allImages / 255
	
	window = np.ones([wSize, wSize])
	window = window / np.sum(window)
	window3D = np.ones((wSize, wSize, 3), dtype=np.float)
	window3D = window3D / np.sum(window3D)

	xIdxMax = allImages.shape[1] - wSize + 1
	yIdxMax = allImages.shape[2] - wSize + 1
	refIdx = selectRef(allImages, exposureThres)

	numExd = 2 * allImages.shape[0] - 1
	temp = list(allImages.shape)
	temp[0] = numExd
	allImagesExd = np.zeros(temp, np.float)
	allImagesExd[:allImages.shape[0]] = allImages

	count = 0
	for i in range(allImages.shape[0]):
		if i != refIdx:
			temp = hist_norm(allImagesExd[refIdx], allImagesExd[i])
			temp = np.where(temp < 0, 0, temp)
			temp = np.where(temp > 1, 1, temp)
			allImagesExd[allImages.shape[0] + count] = temp
			count += 1

	gMu = np.zeros([numExd, xIdxMax, yIdxMax])
	for i in range(numExd):
		gMu[i] = np.ones([xIdxMax, yIdxMax]) * np.mean(allImagesExd[i])

	temp = np.zeros([xIdxMax, yIdxMax, allImages.shape[3]])
	lMu = np.zeros([numExd, xIdxMax, yIdxMax])
	lMuSq = np.zeros([numExd, xIdxMax, yIdxMax])
	for i in range(numExd):
		for j in range(allImages.shape[3]):
			temp[:, :, j] = convolution2d(allImagesExd[i, :, :, j], window)
			print(i, j)
		lMu[i] = np.mean(temp, axis=2)
		lMuSq[i] = lMu[i] * lMu[i]

	sigmaSq = np.zeros([numExd, xIdxMax, yIdxMax])
	for i in range(numExd):
		for j in range(allImages.shape[3]):
			temp[:, :, j] = convolution2d(
				allImagesExd[i, :, :, j] * allImagesExd[i, :, :, j], window) - lMuSq[i]
		sigmaSq[j] = np.mean(temp, axis=2)
	sigma = np.sqrt(np.where(sigmaSq < 0, 0, sigmaSq))
	ed = sigma * np.sqrt(wSize * wSize * allImages.shape[2]) + 1e-3

	sMap = np.zeros([allImages.shape[0], allImages.shape[0], xIdxMax, yIdxMax])
	for i in range(allImages.shape[0]):
		for j in range(i + 1, allImages.shape[0]):
			crossMu = lMu[i] * lMu[j]
			crossSigma = np.squeeze(convolution3d(
				allImagesExd[i]*allImagesExd[j], window3D)) - crossMu
			sMap[i][j] = (crossSigma + C) / (sigma[i] * sigma[j] + C)
	sMap = np.where(sMap < 0, 0, sMap)
	
	sRefMap = np.squeeze(sMap[refIdx]) + sMap[:, refIdx]
	sRefMap[refIdx] = np.ones([xIdxMax, yIdxMax])
	sRefMap = np.where(sRefMap <= structureThres, 0, sRefMap)
	sRefMap = np.where(sRefMap > structureThres, 1, sRefMap)
	muIdxMap = (lMu[refIdx] < exposureThres) | (lMu[refIdx] > 1 - exposureThres)
	sRefMap = np.where(muIdxMap, 1, sRefMap)
	se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(wSize,wSize))
	for i in range(allImages.shape[0]):
		sRefMap[i] = cv2.morphologyEx(sRefMap[i], cv2.MORPH_OPEN, se)

	iRefMap = imfConsistenct(lMu[:allImages.shape[0]], refIdx,consistencyThres)

	cMap = sRefMap * iRefMap

	cMapExd = np.zeros([numExd, xIdxMax, yIdxMax])

	cMapExd[:allImages.shape[0]] = cMap
	count = 0
	for i in range(allImages.shape[0]):
		if i != refIdx:
			cMapExd[count + allImages.shape[0]] = 1 - cMapExd[i]
			count = count+1
	
	muMap = np.exp(-0.5 * (np.power(gMu - 0.5, 2)/np.power(gSig, 2) + np.power(lMu - 0.5, 2) / np.power(lSig, 2)))
	muMap = muMap * cMapExd
	normalizer = np.sum(muMap, axis = 0)
	muMap = muMap / normalizer

	sMap = np.power(ed, p)
	sMap = sMap * cMapExd + 0.001
	normalizer = np.sum(sMap, axis=0)
	sMap = sMap / normalizer

	maxEd = ed * cMapExd
	maxEd = np.max(maxEd, axis = 0)

	indM = np.zeros([allImages.shape[0], xIdxMax, yIdxMax], dtype=np.uint8)
	indM[refIdx] = refIdx
	for i in range(allImages.shape[0]):
		if i < refIdx:
			indM[i] = cMapExd[i] * i + cMapExd[i + allImages.shape[0]] * (i + allImages.shape[0])
		elif i > refIdx:
			indM[i] = cMapExd[i] * i + cMapExd[i + allImages.shape[0] - 1] * (i + allImages.shape[0] - 1)

	fI = np.zeros(allImages.shape[1:])
	countMap = np.zeros(allImages.shape[1:])
	countWindow = np.ones([wSize, wSize, allImages.shape[3]])
	xIdx = list(range(0, xIdxMax, stepSize))
	xIdx += list(range(xIdx[-1] +1, xIdxMax))
	yIdx = list(range(0, yIdxMax, stepSize))
	yIdx += list(range(yIdx[-1] +1, yIdxMax))
	offset = wSize - 1
	blocks = np.zeros([allImages.shape[0], wSize, wSize, allImages.shape[3]])
	for i in xIdx:
		for j in yIdx:
			for k in range(allImages.shape[0]):
				blocks[k] = allImagesExd[indM[k][i][j], i:i+wSize, j:j+wSize]
			rBlock = np.zeros([wSize, wSize, allImages.shape[3]])
			for k in range(allImages.shape[0]):
				rBlock = rBlock + sMap[k][i][j] * (blocks[k] - lMu[k][i][j]) / ed[k][i][j]
			if np.linalg.norm(rBlock) > 0:
				rBlock = rBlock / np.linalg.norm(rBlock) * maxEd[i][j]
			rBlock = rBlock + np.sum(muMap[:, i, k] * lMu[:, i, j])
			fI[i:i+wSize, j:j+wSize] = fI[i:i+wSize, j:j+wSize] + rBlock
			countMap[i:i+wSize, j:j+wSize] = countMap[i:i+wSize, j:j+wSize] + countWindow
	fI = fI / countMap
	fI = np.where(fI > 1, 1, fI)
	fI = np.where(fI < 0, 0, fI)
	return fI

def printHDR(fileName, energys):
	f = open(fileName + ".hdr", "wb")
	f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
	f.write(bytes(
		"-Y {0} +X {1}\n".format(energys.shape[0], energys.shape[1]), encoding='utf8'))

	brightest = np.maximum(np.maximum(
		energys[..., 0], energys[..., 1]), energys[..., 2])
	mantissa = np.zeros_like(brightest)
	exponent = np.zeros_like(brightest)
	np.frexp(brightest, mantissa, exponent)
	scaled_mantissa = mantissa * 256.0 / brightest
	rgbe = np.zeros((energys.shape[0], energys.shape[1], 4), dtype=np.uint8)
	rgbe[..., 0:3] = np.around(energys[..., 0:3] * scaled_mantissa[..., None])
	rgbe[..., 3] = np.around(exponent + 128)
	rgbe.flatten().tofile(f)
	f.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-j", "--jsonPath", type=str,
						help="The json file that store information about images", default="")
	parser.add_argument("-s", "--smooth", type=float,
						help="The weight of smooth parameter", default=1)
	parser.add_argument("-p", "--pixel", type=int,
						help="The number of sample for each g(x)", default=None)
	parser.add_argument("-spd", "--spdMef", action='store_true', 
						help="Using SPD-MEF")
	parser.add_argument("-ngr", "--notGhostRemoval", action='store_false', 
						help="Not Using Ghost removal")
	args = parser.parse_args()

	start_time = time.time()

	allImages, ln_ts = readJson(args.jsonPath)

	if args.spdMef:
		outputs = spdMef(allImages, wSize = 21)
		outputs = np.clip(outputs, 0, 1)
		outputs = outputs * 255
	else:
		outputs, g_Zs = hdr(allImages, ln_ts, args.smooth, args.pixel, args.notGhostRemoval)

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
