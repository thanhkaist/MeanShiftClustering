# MeanShiftClusstering
Implement mean shift clustering  from numpy + toy example with sklearn library

- mean-shift.py: numpy implementation for data with 2 dimension

- mean-shift-np.py: numpy implementation for data with 3 dimension  

- mean-shift-sklearn.py: toy example using sklearn.MeanShift

![result1](fig/3class1.png)

![result2](fig/mean-shift-plot.png)


Statistic: 

|      |      mean-shift-np      | mean-shift-sklean                   |
|:----:|:-----------------------:|-------------------------------------|
| Time | 30.02 s                 | 0.5 s                               |
| Note | No use matrix operator  | Auto bandwidth Parallel with n_jobs |

