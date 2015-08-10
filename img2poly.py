import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull

def main():	
	img_path = sys.argv[1]
	n_points = int(sys.argv[2])
	# validate inputs (check if we have enough points in critical places!)

	img = cv2.imread(img_path, 0)
	canny = cv2.Canny(img, 100, 200)

	###plt.imshow(canny, cmap = 'gray')
	#plt.show()

	edges = np.transpose(np.nonzero(canny))

	np.random.shuffle(edges)
	points = edges[:n_points]
	
	img_points = np.zeros((len(canny), len(canny[0])))
	img_points[zip(*points)] = 255
	plt.imshow(img_points, cmap = 'gray')
	plt.show()

	points=np.vstack((points,np.array([
		[0,0],
		[0,np.shape(canny)[1] - 1],
		[np.shape(canny)[0] - 1,0],
		[np.shape(canny)[0] - 1,np.shape(canny)[1] - 1]
	])))
	
	points=points[:,[1,0]]
	tri=Delaunay(points)

	plt.subplot(121), plt.triplot(points[:,0], (-1)*points[:,1], tri.simplices.copy())
	#plt.plot(points[:,0], points[:,1], 'o')
	###plt.show()

	k = tri.simplices[0]
	v1 = points[k[0]]
	v2 = points[k[1]]
	v3 = points[k[2]]
	a_min = min(v1[0], v2[0], v3[0])
	a_max = max(v1[0], v2[0], v3[0])
	b_min = min(v1[1], v2[1], v3[1])
	b_max = max(v1[1], v2[1], v3[1])

	print v1, v2, v3
	l1 = [v1, v2, v3]
	hull1 = ConvexHull(l1)

	for i in range(a_min, a_max + 1):
		for j in range(b_min, b_max + 1):
			l1.append(np.array([i, j]))
			hull2 = ConvexHull(l1)
			if len(hull1.vertices) == len(hull2.vertices):
				img_points[j][i] = 255
			l1.pop()
			
	plt.subplot(122), plt.imshow(img_points, cmap = 'gray')
	plt.show()

if __name__ == "__main__":
	main()