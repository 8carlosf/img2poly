import sys
import cv2
import numpy as np
import random as rand
import matplotlib.pyplot as plt

def main():	
    img_path = sys.argv[1]
    n_points = int(sys.argv[2])
    # validate inputs (check if we have enough points in critical places!)

    img = cv2.imread(img_path, 0)
    canny = cv2.Canny(img, 100, 200)

    plt.imshow(canny, cmap = 'gray')
    #plt.show()

    edges = np.transpose(np.nonzero(canny))

    np.random.shuffle(edges)
    points = edges[:n_points]

    img_points = np.zeros((len(canny), len(canny[0])))
    img_points[zip(*points)] = 255
    plt.imshow(img_points, cmap = 'gray')
    plt.show()

if __name__ == "__main__":
    main()
