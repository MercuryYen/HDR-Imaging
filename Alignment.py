import numpy as np
from numba import njit
import argparse
import time
from PIL import Image
import math

from readImage import readJson


@njit
def halfBitmap(image):
	newImage = np.zeros((image.shape[0]//2, image.shape[1]//2), dtype=np.uint8)
	for i in range(newImage.shape[0]):
		for j in range(newImage.shape[1]):
			newImage[i][j] = (image[2 * i][2 * j] + image[2*i+1]
							  [2*j] + image[2*i][2*j+1] + image[2*i+1][2*j+1])/4
	return newImage

# return black white mask, and mask


def toMask(image, threshold=0.17):
	sortedPixels = np.sort(image, axis=None)
	threshold = math.floor(len(sortedPixels) * threshold)
	return image > sortedPixels[len(sortedPixels) // 2], np.logical_or(image <= sortedPixels[threshold], image >= sortedPixels[-threshold])


def getShift(imageRef, imageTar, shiftDeep=3):
	shift = [0, 0]
	if shiftDeep > 0:
		shift = getShift(halfBitmap(imageRef),
						 halfBitmap(imageTar), shiftDeep - 1)
		shift = [x * 2 for x in shift]

	bwMaskRef, andMaskRef = toMask(imageRef)

	outputShift = shift

	minErr = imageRef.shape[0] * imageRef.shape[1]
	for y in range(-1, 2):
		for x in range(-1, 2):
			yShift = shift[1] + y
			xShift = shift[0] + x
			testImage = np.roll(imageTar, yShift, axis=0)
			testImage = np.roll(testImage, xShift, axis=1)
			bwMask, andMask = toMask(testImage)

			err = np.count_nonzero((bwMask ^ bwMaskRef) & andMaskRef)
			if err < minErr:
				outputShift = [shift[0] + xShift, shift[1] + yShift]
				minErr = err

	return outputShift


@njit
def _toGray(image, grayImage):
	for y in range(grayImage.shape[0]):
		for x in range(grayImage.shape[1]):
			pixel = image[y][x]
			grayImage[y][x] = (pixel[0] * 54 + pixel[1] *
							   183 + pixel[2] * 19) / 256
	return grayImage


def toGray(image):
	grayImage = np.zeros(image.shape[:2], np.float)
	return _toGray(image, grayImage)


def alignment(images, shiftDepth=5):
	imageRef = images[len(images) // 2]
	for i in range(len(images)):
		shift = getShift(toGray(imageRef), toGray(images[i]), shiftDepth)
		images[i] = np.roll(
			np.roll(images[i], shift[1], axis=0), shift[0], axis=1)

		if shift[0] > 0:
			images[i][:, :shift[0]] = (0, 0, 255)
		elif shift[0] < 0:
			images[i][:, shift[0]:] = (0, 0, 255)
		if shift[1] > 0:
			images[i][:shift[1]] = (0, 0, 255)
		elif shift[1] < 0:
			images[i][shift[1]:] = (0, 0, 255)
		
		print(f"{i+1} / {len(images)}")

	return images


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-j", "--jsonPath", type=str,
						help="The json file that store information about images.", default="")
	parser.add_argument("-d", "--shiftDepth", type=int,
						help="The depth of shift recursive.", default=5)

	args = parser.parse_args()

	start_time = time.time()
	allImages, ln_ts = readJson(args.jsonPath)

	allImages = alignment(allImages, args.shiftDepth)

	print(f"Spend {time.time() - start_time} sec")

	for i in range(len(allImages)):
		image = Image.fromarray(allImages[i])
		image.save(f"temp{i}.png")
