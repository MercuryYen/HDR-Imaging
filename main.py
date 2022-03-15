from hdr import hdr
from alignment import alignment
from readImage import readJson

import numpy as np

from PIL import Image

import argparse

import time

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-j", "--jsonPath", type=str,
						help="The json file that store information about images.", default="")
	parser.add_argument("-s", "--smooth", type=float,
						help="The weight of smooth parameter", default=1)
	parser.add_argument("-p", "--pixel", type=int,
						help="The number of sample for each g(x)", default=None)
	parser.add_argument("-d", "--shiftDepth", type=int,
						help="The depth of shift recursive.", default=5)

	args = parser.parse_args()

	start_time = time.time()
	allImages, ln_ts = readJson(args.jsonPath)

	allImages = alignment(allImages, args.shiftDepth)
	outputs, g_Zs = hdr(allImages,ln_ts, args.smooth, args.pixel)

	print(f"Spend {time.time() - start_time} sec")

	# display
	maxVal = max([np.amax(np.log(output)) for output in outputs])
	minVal = min([np.amin(np.log(output)) for output in outputs])
	print(min([np.amin(np.log(output)) for output in outputs]))
	print(max([np.amax(np.log(output)) for output in outputs]))
	outputs = [(np.log(output) - minVal) * 255 / (maxVal - minVal)
			   for output in outputs]

	output_image = np.zeros(
		(outputs[0].shape[0], outputs[0].shape[1], 3), 'uint8')
	for i in range(3):
		output_image[..., i] = outputs[i]
	image = Image.fromarray(output_image)
	image.save("temp.png")
	image.show()

	print(f"Spend {time.time() - start_time} sec")
