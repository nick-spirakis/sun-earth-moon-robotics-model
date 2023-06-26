from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

#====================================================

E = 10   # round of 10 decimal points



a1 = 1   # length of link a1 in cm
a2 = 39  # length of link a2 in cm distance of sun to earth
a3 = 1   # length of link a3 in cm
a4 = 20  #distance between moon and earth

#change these to rotate earth and moon
T1= 0  # Theta 1 The Earth
T2= 0  # Theta 2 The Moon

T1 = (T1/180.0)*np.pi  # Theta 1 angle in radians
T2 = (T2/180.0)*np.pi  # Theta 2 angle in radians

# rotational matrix, 2 rotational joints, both around z axis

R0_1 = np.array([
    [np.cos(T1), -np.sin(T1), 0],
    [np.sin(T1),  np.cos(T1), 0], 
    [         0,           0, 1]
])
print("RM R0_1:")
print(R0_1)
print()

R1_2 = np.array(
    [[np.cos(T2), -np.sin(T2), 0],
     [np.sin(T2),  np.cos(T2), 0], 
     [         0,           0, 1]
])
print("RM R1_2:")
print(R1_2)
print()

R0_2 = np.dot(R0_1, R1_2)
R0_2 = np.ndarray.round(R0_2, E)
print("RM R0_2:")
print(R0_2)
print()


# -------------------------------------------------------------------------------


# Displacement Vectors
d0_1 = [a2 * np.cos(T1), a2 * (np.sin(T1)), a1]
d0_1 = np.array(d0_1)
d0_1 = d0_1.reshape(3, 1)
d0_1 = np.ndarray.round(d0_1, E)
print("DV d0_1:")
print(d0_1)
print()

d1_2 = [a4 * np.cos(T2), a4 * (np.sin(T2)), a3]
d1_2 = np.array(d1_2)
d1_2 = d1_2.reshape(3, 1)
d1_2 = np.ndarray.round(d1_2, E)
print("DV d1_2:")
print(d1_2)


#----------------------------------------------------------------------
# homogeneous 
H0_1 = np.concatenate((R0_1, d0_1), 1)
H0_1 = H0_1.reshape(3, 4)

H0_1 = np.concatenate((H0_1,[[0, 0, 0, 1]]), 0)
H0_1 = np.ndarray.round(H0_1, E)
print("HTM H0_1:")
print(H0_1)
print()

H1_2 = np.concatenate((R1_2, d1_2), 1)
H1_2 = np.concatenate((H1_2,[[0, 0, 0, 1]]), 0)
H1_2 = np.ndarray.round(H1_2, E)
print("HTM H1_2:")
print(H1_2)
print()

H0_2 = H0_1.dot(H1_2)
H0_2 = np.ndarray.round(H0_2, E)
print("HTM H0_2:")
H0_2 = np.ndarray.round(H0_2, E)
print(H0_2)
print()



print("")

#V = [40, 0, 2]
#moon V
V1 = [H0_2[0][3], H0_2[1][3], H0_2[2][3]]
print("V1:", V1)

#earth V
V = [H0_1[0][3], H0_1[1][3], H0_1[2][3]]
print("V:", V)

U = np.dot(V1, R0_2)
U = np.ndarray.round(U, E)

print("U: ")
print(U)

# -------------------------------------------


# Data Visualization
fig = plt.figure("Location of the moon", figsize=(12, 9)) #12, 9
ax = fig.gca(projection='3d')


ax.set_aspect("auto")


plt.title("Unit Sphere Chart"); 

# draw a point

ax.scatter([0], [0], [0], color="orange", s=100)
ax.scatter([50], [0], [0], color="k", s=0)
ax.scatter([-50], [0], [0], color="k", s=0)
ax.scatter([0], [50], [0], color="k", s=0)
ax.scatter([0], [-50], [0], color="k", s=0)
ax.scatter([0], [0], [50], color="k", s=0)
ax.scatter([0], [0], [-50], color="k", s=0)


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# draw a vector
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


sun = [0, 0, 0]
# the Sun
ax.text(0.05, 0.05, 5.05,  '%s' % "Sun"+str(sun), size=16, zorder=1,  color='orange')


# the earth, color blue
ax.text(V[0]+0.05,V[1]+0.05,V[2]+5.05,  '%s' % "Earth"+str(V), size=16, zorder=1,  color='b')

ht = Arrow3D([0, V[0]], [0, V[1]], [0, V[2]], mutation_scale=20,
            lw=3, arrowstyle="-|>", color="b")
# this adds blue line
ax.add_artist(ht)



# the moon, color black
ax.text(V1[0]+0.05,V1[1]+0.05,V1[2]+4.05,  '%s' % "Moon"+str(V1), size=16, zorder=1,  color='k')

hb = Arrow3D([V[0], V1[0]], [V[1], V1[1]], [V[2], V1[2]], mutation_scale=20,
            lw=3, arrowstyle="-|>", color="k")
#vi to U
# this adds black line
ax.add_artist(hb)




plt.show()