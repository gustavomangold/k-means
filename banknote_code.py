# really simple example
# banknote authentication through k-means classification
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from sklearn.cluster import KMeans as kmeans

data = pd.read_csv('auth_note_data.csv')
data.sort_values('V1', inplace=True)

v1 = data['V1']
v2 = data['V2']

mean = np.mean(data, 0)
std_dev = np.std(data, 0)
v1_v2 = np.column_stack((v1, v2))

km = kmeans(n_clusters = 2)
km_res = km.fit_predict(data)
clusters = km.cluster_centers_

print('Mean of V1:', mean[0], ', Std. Dev. of V1:', std_dev[0])
print('Mean of V2:', mean[1], ', Std. Dev. of V2:', std_dev[1])

#this part can be used to generate an 'error' ellipse around the mean, with two standard deviations
#ellipse = patches.Ellipse([mean[0], mean[1]], std_dev[0]*2, std_dev[1]*2, alpha = 0.42, color = (1, 0, 0, 1))
#plt.scatter(v1_new_1, v2_new_1, alpha = 0.4, s=50)
#plt.scatter(v1_new_2, v2_new_2, alpha = 0.4, s=50)
#graph.add_patch(ellipse)

x,y = data[km_res == 0], data[km_res==1]
plt.scatter(x.iloc[:,0], x.iloc[:,1], alpha = 0.4)
plt.scatter(y.iloc[:,0], y.iloc[:,1], alpha = 0.4)
plt.scatter(clusters[:,0], clusters [:,1], s=500, alpha=0.5, color = (1,0,0,1))
plt.savefig('k-means.png')
plt.show()
