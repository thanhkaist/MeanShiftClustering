import numpy as np
import random

STOP_THRESHOLD = 1e-4
CLUSTER_THRESHOLD = 1e-1

def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def gaussian_kernel(distance, bandwidth):
    return (1 / (bandwidth * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((distance / bandwidth)) ** 2)

class MeanShift(object):
    def __init__(self, kernel=gaussian_kernel):
        self.kernel = kernel

    def fit(self, points, kernel_bandwidth):

        shift_points = np.array(points)
        shifting = [True] * points.shape[0]

        while True:
            max_dist = 0
            for i in range(0, len(shift_points)):
                if not shifting[i]:
                    continue
                p_shift_init = shift_points[i].copy()
                shift_points[i] = self._shift_point(shift_points[i], points, kernel_bandwidth)
                dist = distance(shift_points[i], p_shift_init)
                max_dist = max(max_dist, dist)
                shifting[i] = dist > STOP_THRESHOLD

            if(max_dist < STOP_THRESHOLD):
                break
        cluster_ids, centers = self._cluster_points(shift_points.tolist())
        self.labels = np.array(cluster_ids)
        self.centers = np.array(centers)
        return shift_points, cluster_ids

    def _shift_point(self, point, points, kernel_bandwidth):
        shift_x = 0.0
        shift_y = 0.0
        shift_z = 0.0
        scale = 0.0
        for p in points:
            dist = distance(point, p)
            weight = self.kernel(dist, kernel_bandwidth)
            shift_x += p[0] * weight
            shift_y += p[1] * weight
            shift_z += p[2] * weight
            scale += weight
        shift_x = shift_x / scale
        shift_y = shift_y / scale
        shift_z = shift_z / scale
        return [shift_x, shift_y,shift_z]

    def _cluster_points(self, points):
        cluster_ids = []
        cluster_idx = 0
        cluster_centers = []

        for i, point in enumerate(points):
            if(len(cluster_ids) == 0):
                cluster_ids.append(cluster_idx)
                cluster_centers.append(point)
                cluster_idx += 1
            else:
                for center in cluster_centers:
                    dist = distance(point, center)
                    if(dist < CLUSTER_THRESHOLD):
                        cluster_ids.append(cluster_centers.index(center))
                if(len(cluster_ids) < i + 1):
                    cluster_ids.append(cluster_idx)
                    cluster_centers.append(point)
                    cluster_idx += 1
        return cluster_ids, cluster_centers


from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
import time 

style.use("ggplot")

def colors(n):
  ret = []
  for i in range(n):
    ret.append((random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
  return ret

def main():
    centers = [[1,1,1],[5,5,5],[3,10,10]]

    X, y = make_blobs(n_samples = [100,100,100], centers = centers, cluster_std = 0.5)
    ms = MeanShift() 
    
    begin_time = time.time()
    ms.fit(X, kernel_bandwidth=0.5)
    end_time = time.time()

    print("Total time (s)", end_time- begin_time)

    cluster_centers = ms.centers
    labels = ms.labels

    print(cluster_centers)
    n_clusters_ = len(np.unique(labels))

    print("Number of estimated cluster:", n_clusters_)

    colors = 10*['r','g','b','c','k','y','m']

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    for i in range(len(X)):
        ax.scatter(X[i][0],X[i][1],X[i][2], c=colors[labels[i]],marker='o')

    ax.scatter(cluster_centers[:,0],cluster_centers[:,1],cluster_centers[:,2],marker="x",color='k',s=150, linewidths =5, zorder =10)
    plt.show()
    

    

if __name__ == '__main__':
    main()