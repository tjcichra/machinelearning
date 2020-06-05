#!/usr/bin/python3
from mnist import MNIST
import random

mndata = MNIST('samples')
images, labels = mndata.load_training()

while True:
	index = random.randrange(0, len(images))
	print("Image: " + str(labels[index]))
	for pixel in images[index]:
		print(str(pixel), end = " ")
	print("")
	input();
