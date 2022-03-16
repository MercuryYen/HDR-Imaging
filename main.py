from hdr import hdr
from alignment import alignment
from readImage import readJson
from tone import globalOperator

import numpy as np

from PIL import Image

import argparse

import time

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
	energys, g_Zs = hdr(allImages,ln_ts)

	print(f"Spend {time.time() - start_time} sec")

	# display
	luminances = globalOperator(energys, 0.18, 100000000)

	image = Image.fromarray(np.around(luminances * 255).astype(np.uint8))
	image.save("temp.png")
	image.show()

	print(f"Spend {time.time() - start_time} sec")
