In Unsupervised Learning the algorithm discovers patterns in unlabelled data. Aiming to find hidden insights, group data, or simplify complex datasets.

K-Means Clustering is asking how we can group similar points together

We want to split our data into K clusters

Each cluster has a centroid (the mean point)

Our algorithm will repeat the steps:
1. Assign points to nearest centroid
2. Update the centroids to the mean of the assigned points

We want to minimize the distance between data points and their assigned cluster centroids

$J = \Sigma ||x_i - \mu_{c_i}||$

$x_i$ is the data point \
$\mu_{c_i}$ is the centroid of the assigned cluster