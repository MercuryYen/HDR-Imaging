import numpy as np
from hdr import hdr
from alignment import toGrey
from readImage import readJson

import argparse
import time
from PIL import Image

def globalOperator(energys, alpha = 0.18, Lwhite = 1.0):
	delta = 0.000001

	greyImage = toGrey(energys)
	Lw = np.exp(np.sum(np.log(greyImage + delta)) / greyImage.size)

	print(f"Lw: {Lw}")

	Lm = alpha * energys / Lw

	Ld = (Lm * (1 + Lm / Lwhite ** 2)) / (1 + Lm)

	return Ld

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-j", "--jsonPath", type=str,
						help="The json file that store information about images", default="")
	parser.add_argument("-s", "--smooth", type=float,
						help="The weight of smooth parameter", default=500)
	parser.add_argument("-p", "--pixel", type=int,
						help="The number of sample for each g(x)", default=500)
	parser.add_argument("-a", "--alpha", type=float,
						help="alpha", default=0.18)
	parser.add_argument("-l", "--Lwhite", type=float,
						help="alpha", default=100000)
	args = parser.parse_args()

	start_time = time.time()

	allImages, ln_ts = readJson(args.jsonPath)
	energys, g_Zs = hdr(allImages, ln_ts, args.smooth, args.pixel)
	#luminances = globalOperator(allImages[4] / 255, 0.72, 1)
	luminances = globalOperator(energys, args.alpha, args.Lwhite)
	# outputs = [energy*luminance for energy, luminance in zip(energys,luminances)]
	outputs = luminances

	print(f"Spend {time.time() - start_time} sec")

	# display
	# maxVal = np.amax(outputs)
	# minVal = np.amin(outputs)

	# outputs = (outputs - minVal) * 255 / (maxVal - minVal)
	image = Image.fromarray(np.around(outputs * 255).astype(np.uint8))
	image.show()
	image.save("temp.png")