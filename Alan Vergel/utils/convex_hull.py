import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt



class CHull(ConvexHull):
    
    def __init__(self, points):
        ConvexHull.__init__(self, points)

    def centrum(self):

        c = []
        for i in range(self.points.shape[1]):
            c.append(np.mean(self.points[self.vertices,i]))

        return c

    def show(self):
        #Plot convex hull
        for simplex in self.simplices:
            plt.plot(self.points[simplex, 0], self.points[simplex, 1], 'k-')

        

    def show_centroid(self):
        #Show convex hull.
        self.show()

        c = self.centrum()

        #Plot centroid
        plt.plot(c[0], c[1],'x',ms=20)
        plt.show()
        



if __name__ == "__main__":
    points = np.random.rand(30, 2)   # 30 random points in 2-D
    #hull = ConvexHull(points)

    #Get centroid
    #cx = np.mean(hull.points[hull.vertices,0])
    #cy = np.mean(hull.points[hull.vertices,1])

    hull = CHull(points)
    c = hull.centrum()
    print("Centroid:", c)

    hull.show_centroid()


    
    
