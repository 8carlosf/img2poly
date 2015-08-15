import sys
import numpy as np
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

def checkNeighbours(new_point, grid, min_dist, a, b):
	k_min = new_point[0] - min_dist
	k_max = new_point[0] + min_dist
	k2_min = new_point[1] - min_dist
	k2_max = new_point[1] + min_dist
	if k_min < 0:
		k_min = 0
	if k_max >= a:
		k_max = a - 1
	if k2_min < 0:
		k2_min = 0
	if k2_max >= b:
		k2_max = b - 1

	for i in range(k_min, k_max + 1):
		for j in range(k2_min, k2_max + 1):
			if grid[i][j] == 1 and np.sqrt(np.square(new_point[0] - i) + np.square(new_point[1] - j)) < min_dist:
				return False

	return True

def poisson(min_dist, a, b, n_points):
	randomQueue = []
	poisson_points = []
	grid = np.zeros((a, b))

	first = (randint(0, a - 1), randint(0, b - 1))
	poisson_points += [first]
	randomQueue += [first]
	grid[first[0]][first[1]] = 1

	while len(randomQueue) != 0:
		print len(randomQueue)
		point = randomQueue.pop(randint(0, len(randomQueue) - 1))
		for i in range(0, n_points):
			new_point = gen_random_poisson_point(point, min_dist, a, b)
			if grid[new_point[0]][new_point[1]] == 0 and checkNeighbours(new_point, grid, min_dist, a, b) == True:
				randomQueue += [new_point]
				poisson_points += [new_point]
				grid[new_point[0]][new_point[1]] = 1

	return poisson_points


def main():
	img_path = sys.argv[1]
	n_points = int(sys.argv[2])
	n_upoints = int(sys.argv[3])
	min_dist = int(sys.argv[4])
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
	uni_points = poisson(min_dist, len(canny), len(canny[0]), n_upoints)

	np.random.shuffle(edges)
	points = edges[:n_points] + uni_points

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
		in_tri[points[t,0], points[t,1]] = 0

	plt.imshow(img_points)
	plt.show()

if __name__ == "__main__":
	main()
