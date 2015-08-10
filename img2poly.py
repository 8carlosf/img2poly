import sys
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from skimage import feature, io, data, color
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt
from random import randint

def main():	
    img_path = sys.argv[1]
    n_points = int(sys.argv[2])
    # validate inputs (check if we have enough points in critical places!)

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
    ###plt.imshow(canny, cmap = 'gray')
    #plt.show()

    #edges = np.transpose(np.nonzero(canny))
    uni_points = []
    block_size = 100
    for i in range(0, len(canny)-block_size, block_size):
        for j in range(0, len(canny[0])-block_size, block_size):
            uni_points += [(i+randint(0, block_size), j+randint(0, block_size))]

    np.random.shuffle(edges)
    np.random.shuffle(uni_points)
    
    points = edges[:n_points] + uni_points[:n_points]

    img_points = np.zeros((len(canny), len(canny[0])))
    #img_points[canny] = 255
    for x,y in points:
        img_points[x,y] = 255
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

    plt.triplot(points[:,0], (-1)*points[:,1], tri.simplices.copy())
    #plt.plot(points[:,0], points[:,1], 'o')
    #plt.show()

    #img_points = np.zeros((len(canny), len(canny[0])))
    img_points = img
    in_tri = np.zeros((len(canny), len(canny[0])))
    for t in tri.simplices:
        in_tri[points[t,0], points[t,1]] = 255
        chull = convex_hull_image(in_tri)
        img_points[chull, 0] = np.mean(img_points[chull, 0])
        img_points[chull, 1] = np.mean(img_points[chull, 1])
        img_points[chull, 2] = np.mean(img_points[chull, 2])
        in_tri[points[t,0], points[t,1]] = 0

    plt.imshow(img_points)
    plt.show()

    '''
    k = tri.simplices[0]
    v1 = points[k[0]]
    v2 = points[k[1]]
    v3 = points[k[2]]
    a_min = min(v1[0], v2[0], v3[0])
    a_max = max(v1[0], v2[0], v3[0])
    b_min = min(v1[1], v2[1], v3[1])
    b_max = max(v1[1], v2[1], v3[1])

    print (v1, v2, v3)
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
    '''

if __name__ == "__main__":
    main()
