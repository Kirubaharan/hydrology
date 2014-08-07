__author__ = 'kiruba'
from numpy import array
h = array([[4., 5., 6., 7.],
          [4., 0., 0., 7,],
          [4., 0., 0., 7.],
          [4., 5., 6., 7.]])
dummy = h.shape
nrow = dummy[0]
ncol = dummy[1]

print 'Head matrix is a ', nrow, 'by', ncol, 'matrix'

ni = 1
conv_crit = 1e-3
converged = False
w = 1.1

while (not converged):
    # max_err = 0
    for r in range(1, nrow-1):
        for c in range(1, ncol-1):
            h_old = h[r, c]
            print h_old
            h[r, c] = (h[r-1, c] + h[r+1, c] + h[r, c-1] + h[r, c+1])/4
            print h[r, c]
            c_1 = h[r, c] - h_old
            print c_1
            h[r, c] += (w * c_1)
            print h[r, c]
            diff = h[r, c] - h_old
            print diff
            if diff < conv_crit:
                converged = True
    ni = ni +1

# while (not converged):
#     max_err = 0
#     for r in range(1, nrow-1):
#         for c in range(1, ncol-1):
#             h_old = h[r, c]
#             h[r, c] = (h[r-1, c] + h[r+1, c] + h[r, c-1] + h[r, c+1])/4
#             diff = h[r, c] - h_old
#             if (diff > max_err):
#                 max_err = diff
#     if (max_err < conv_crit):
#         converged = True
#     ni += 1
print 'Number of iterations = ', ni-1
print h