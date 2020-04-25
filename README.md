# MeanShiftClusstering
Implement mean shift clustering  from numpy + toy example with sklearn library

- mean-shift.py: numpy implementation for data with 2 dimension

- mean-shift-np.py: numpy implementation for data with 3 dimension  

- mean-shift-sklearn.py: toy example using sklearn.MeanShift

![result1](fig/3class1.png)

![result2](fig/mean-shift-plot.png)


Statistic: 

|      |mean-shift-np (300p)     | mean-shift-sklean(300p)             |
|:----:|:-----------------------:|-------------------------------------|
| Time | 30.02 s                 | 0.5 s                               |
| Note | No use matrix operator  | Auto bandwidth\\ Parallel with n_jobs |

mean-shift-pytorch time statistic

| No data | cpus | time    |
|---------|------|---------|
| 300     | 1    | 0.4s    |
| 300     | 8    | 1.46s   |
| 3000    | 1    | 5.7s    |
| 3000    | 8    | 3.55s   |
| 30000   | 1    | 136.87s |
| 30000   | 8    | 73.10s  |

