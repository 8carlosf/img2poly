import sys
import cv2
import numpy
from matplotlib import pyplot as plot
from math import ceil

def main():
	
	img_path = sys.argv[1]
	point_ratio = float(sys.argv[2])

	img = cv2.imread(img_path, 0)
	canny = cv2.Canny(img, 50, 150)

	plot.subplot(121), plot.imshow(canny, cmap = 'gray')

	edges = []
	points = []
	for i, k in enumerate(canny):
		for j, v in enumerate(k):
			if(v == 255):
				edges.append((i, j))

	n_points = int(ceil(len(edges) * point_ratio))
	gap = int(len(edges) / n_points)
	ind = 0

	while(ind < len(edges) and len(points) < n_points):
		points.append(edges[ind])
		ind += gap

if __name__ == "__main__":
	main()