import numpy as np
# a=np.zeros((3,3))
# a[1,1] = 1
# print(a)
# b = np.random.random((3,3))
# print(b)
# # print(a*b)
#
# b[1,[0, 2]] = 5 + b[1,[2, 0]]

# b = np.array([1,-6,0])[:,np.newaxis]
# # b1    = b <= -5
# print(b,b.shape)
# a = np.array([1,2,3])[np.newaxis,:]
# print(a,a.shape)
# c = b - a
# print(c,c.shape)
# print(c*100,c.shape)
# print(np.max(c , axis = 0))
# print(np.max(c*100, axis = 0))
# bb = np.max((c & (c*100)), axis = 0)
# print(bb)

ccc = np.array([[ 0,1,1],[   0,1,0]])
print(ccc.shape)
print(ccc[:,False,True,False])
# print(np.max(ccc , axis = 0))