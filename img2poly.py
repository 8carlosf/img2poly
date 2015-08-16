from __future__ import division
import sys
import numpy as np
from math import ceil
from scipy.spatial import Delaunay, ConvexHull
from skimage import feature, io, data, color
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt
from random import randint, seed, uniform

def gen_random_poisson_point(point, min_dist, a, b):
	while True:
		rand1 = uniform(0, 1)
		rand2 = uniform(0, 1)
		radius = min_dist * (rand1 + 1)
		angle = 2 * np.pi * rand2
		x = int(point[0] + radius * np.cos(angle))
		y = int(point[1] + radius * np.sin(angle))
		if x >= 0 and x < a and y >= 0 and y < b:
			return (x, y) 

def poisson(min_dist, a, b, n_points):
	cell_size = min_dist/np.sqrt(2)
	grid = [[None] * int(ceil(b/cell_size)) for i in range(int(ceil(a/cell_size)))]

	randomQueue = []
	poisson_points = []
	#grid = np.zeros((a, b))

	first = (randint(0, a - 1), randint(0, b - 1))
	poisson_points += [first]
	randomQueue += [first]
	gx, gy = coords2grid(first[0], first[1], cell_size)
	grid[gx][gy] = first

	while len(randomQueue) != 0:
		#print(len(randomQueue))
		point = randomQueue.pop(randint(0, len(randomQueue) - 1))
		for i in range(0, n_points):
			new_point = gen_random_poisson_point(point, min_dist, a, b)
			gx, gy = coords2grid(new_point[0], new_point[1], cell_size)
			if grid[gx][gy] == None and checkNeighbourhood(grid, new_point[0], new_point[1], gx, gy, min_dist, cell_size):
				randomQueue += [new_point]
				poisson_points += [new_point]
				grid[gx][gy] = new_point

	return poisson_points

def coords2grid(x, y, cell_size):
	return (int)(x/cell_size), (int)(y/cell_size)

def checkNeighbourhood(grid, x, y, gx, gy, min_dist, cell_size):
	cellsAround = [[-1, 0], [1, 0], [0, -1], [0, 1], [1, 1], [1, -1], [-1, 1], [-1, -1], [-1, -2], [0, -2], [1, -2], [-1, 2], [0, 2], [1, 2], [-2, -1], [-2, 0], [-2, 1], [2, -1], [2, 0], [2, 1]]
	for cell in cellsAround:
		cx, cy = cell[0] + gx, cell[1] + gy
		if (cx >= 0 and cy >= 0 and cx < len(grid) and cy < len(grid[0])):
				if (grid[cx][cy] != None and np.sqrt(np.square(grid[cx][cy][0] - x) + np.square(grid[cx][cy][1] - y)) < min_dist):
					return False
	return True

def poisson_filter(points, min_dist, width, height):
	cell_size = min_dist/np.sqrt(2)
	grid = [[None] * int(ceil(width/cell_size)) for i in range(int(ceil(height/cell_size)))]

	np.random.shuffle(points)
	for p in list(points):
		x, y = p
		gx, gy = coords2grid(x, y, cell_size)
		if grid[gx][gy] == None and checkNeighbourhood(grid, x, y, gx, gy, min_dist, cell_size):
			grid[gx][gy] = p
		else:
			points.remove(p)

	return points

def main():
	print("Usage:   python3 img2poly.py <image> <min_dist_canny> <min_dist_extra>\n")
	img_path = sys.argv[1]
	min_dist = int(sys.argv[2])
	min_distX = int(sys.argv[3])
	# validate inputs (check if we have enough points in critical places!)
	np.random.seed(8)
	seed(8)

	img = io.imread(img_path)
	img_gray = color.rgb2gray(img)
	canny = feature.canny(img_gray, sigma=3)
	print("canny done")

	edges = []
	for i in range(len(canny)):
		for j in range(len(canny[0])):
			if canny[i][j]:
				edges += [(i, j)]

	print("canny size: ", len(edges))
	#plt.imshow(canny, cmap = 'gray')
	#plt.show()

	#edges = np.transpose(np.nonzero(canny))

	uni_points = []
	'''
	block_size = 100
	for i in range(0, len(canny)-block_size, block_size):
		for j in range(0, len(canny[0])-block_size, block_size):
			uni_points += [(i+randint(0, block_size), j+randint(0, block_size))]
	np.random.shuffle(uni_points)
	'''
	'''for i in range(n_upoints):
		uni_points += [(randint(0, len(canny)-1), randint(0, len(canny[0])-1))]
	'''

	'''
	uni_points = poisson(min_dist, len(canny), len(canny[0]), n_upoints)

	np.random.shuffle(edges)
	'''
	points = poisson_filter(edges, min_dist, len(canny[0]), len(canny))
	print("filtered canny size: ", len(points))
	uni_points = poisson(min_distX, len(canny), len(canny[0]), 16)
	print("generated random poisson points: ", len(uni_points))
	points += uni_points

	#img_points = np.zeros((len(canny), len(canny[0])))
	#img_points[canny] = 255
	#for x,y in points:
	#    img_points[x,y] = 255
	#plt.imshow(img_points, cmap = 'gray')
	#plt.show()

	points=np.vstack((points,np.array([
		[0,0],
		[0,np.shape(canny)[1] - 1],
		[np.shape(canny)[0] - 1,0],
		[np.shape(canny)[0] - 1,np.shape(canny)[1] - 1]
		])))
	#points=points[:,[1,0]]
	#points=np.vstack(points)

	tri=Delaunay(points)

	#plt.triplot(points[:,0], (-1)*points[:,1], tri.simplices.copy())
	#plt.plot(points[:,0], points[:,1], 'o')
	#plt.show()

	#img_points = np.zeros((len(canny), len(canny[0])))
	img_points = img
	in_tri = np.zeros((len(canny), len(canny[0])))

	total = len(tri.simplices)
	print("#tri: ", total)

	count = 0
	for t in tri.simplices:
		count += 1
		if (count % 10 == 0):
			print("%3d%%" % (100*count/total))
		in_tri[points[t,0], points[t,1]] = 255
		chull = convex_hull_image(in_tri)
		
		img_points[chull, 0] = np.mean(img_points[chull, 0])
		img_points[chull, 1] = np.mean(img_points[chull, 1])
		img_points[chull, 2] = np.mean(img_points[chull, 2])
		'''
		img_points[chull, 0] = int(np.mean(np.sqrt(img_points[chull, 0]))**2)
		img_points[chull, 1] = int(np.mean(np.sqrt(img_points[chull, 1]))**2)
		img_points[chull, 2] = int(np.mean(np.sqrt(img_points[chull, 2]))**2)
		'''
		
		in_tri[points[t,0], points[t,1]] = 0

	plt.imshow(img_points)
	plt.show()

if __name__ == "__main__":
	main()
