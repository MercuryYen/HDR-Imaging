import numpy as np
from numba import njit
import argparse
import time
from PIL import Image
import math

from readImage import readJson


@njit
def halfBitmap(image):
	newImage = np.zeros((image.shape[0]//2, image.shape[1]//2) + image.shape[2:], dtype=np.uint8)
	for i in range(newImage.shape[0]):
		for j in range(newImage.shape[1]):
			newImage[i][j] = (image[2 * i][2 * j] + image[2*i+1]
							  [2*j] + image[2*i][2*j+1] + image[2*i+1][2*j+1])/4
	return newImage

# return black white mask, and mask


def toMask(image, threshold=0.4):
	sortedPixels = np.sort(image, axis=None)
	threshold = math.floor(len(sortedPixels) * threshold)
	return image > sortedPixels[len(sortedPixels) // 2], np.logical_or(image <= sortedPixels[threshold], image >= sortedPixels[-threshold])

def clip(image, shift, color):
	if shift[0] > 0:
		image[:, :shift[0]] = color
	elif shift[0] < 0:
		image[:, shift[0]:] = color
	if shift[1] > 0:
		image[:shift[1]] = color
	elif shift[1] < 0:
		image[shift[1]:] = color
	return image

def getShift(imageRef, imageTar, shiftDeep=3):
	shift = [0, 0]
	if shiftDeep > 0:
		shift = getShift(halfBitmap(imageRef),
						 halfBitmap(imageTar), shiftDeep - 1)
		shift = [x * 2 for x in shift]

	bwMaskRef, andMaskRef = toMask(imageRef)
	bwMask, andMask = toMask(imageTar)

	andMaskRef = clip(andMaskRef, [shift[0] - 2, shift[1] - 2], False)
	andMaskRef = clip(andMaskRef, [shift[0] + 2, shift[1] + 2], False)


	outputShift = shift

	minErr = imageRef.shape[0] * imageRef.shape[1]
	for y in range(-1, 2):
		for x in range(-1, 2):
			yShift = shift[1] + y
			xShift = shift[0] + x

			bwMaskTemp = np.roll(bwMask, yShift, axis=0)
			bwMaskTemp = np.roll(bwMaskTemp, xShift, axis=1)
			
			andMaskTemp = np.roll(andMask, yShift, axis=0)
			andMaskTemp = np.roll(andMaskTemp, xShift, axis=1)

			err = np.count_nonzero((bwMaskTemp ^ bwMaskRef) & andMaskRef & andMaskTemp)
			if err < minErr:
				outputShift = [shift[0] + xShift, shift[1] + yShift]
				minErr = err

	return outputShift

def toGrey(image, mode = 0):
	if (mode == 0):
		return image[..., 0] * 0.27 + image[..., 1] * 0.67 + image[..., 2] * 0.06
	return image[..., 0] * 0.211 + image[..., 1] * 0.715 + image[..., 2] * 0.074


def alignment(images, shiftDepth=5):
	middle = len(images) // 2
	newImages = []
	count = 0

	newImages.append(images[0])
	for i in range(len(images) - 1):
		shift = getShift(toGrey(images[i]), toGrey(images[i + 1]), shiftDepth)
		image = np.roll(
			np.roll(images[i + 1], shift[1], axis=0), shift[0], axis=1)

		image = clip(image, shift, (0, 0, 255))

		newImages.append(image)

		count += 1
		print(f"{count} / {len(images)}")

	return newImages


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-j", "--jsonPath", type=str,
						help="The json file that store information about images.", default="")
	parser.add_argument("-d", "--shiftDepth", type=int,
						help="The depth of shift recursive.", default=3)
	parser.add_argument("-i", "--iteration", type=int,
						help="Half image iteration", default=0)

	args = parser.parse_args()

	start_time = time.time()
	allImages, ln_ts = readJson(args.jsonPath)
	newAllImages = []

	if args.iteration > 0:
		for i in range(len(allImages)):
			target = allImages[i]
			for iter in range(args.iteration):
				target = halfBitmap(target)
			newAllImages.append(target)
		allImages = np.array(newAllImages)

	allImages = alignment(allImages, args.shiftDepth)
	
	print(f"Spend {time.time() - start_time} sec")

	for i in range(len(allImages)):
		image = Image.fromarray(allImages[i])
		image.save(f"temp{i}.png")
