import pickle
with open('data.pickle', 'rb') as f:
	NP = pickle.load(f)
#print(NP[69714:69794])
HIST1 = NP[69714:69794]
#data = {Columns: len(HIST1[0]), Rows: len(HIST1)}
row_sums = []
col_sums = []
del_idx = []
row_range = len(HIST1)
row_start = 69714
row_end = 69794
col_start = 0
col_end = len(NP[0])
columns = []
#Similarity = Sum(min(x,y))/Sum(max(x,y)) for x in NP1 and y in NP2
for j in range(0, col_end): #0 to 408 columns
	col_sums.append(0)
	temp = []
	columns.append(temp)

for i in range(0,len(HIST1)):
	row_sums.append(sum(HIST1[i]))
	for j in range(0,col_end):
		columns[j].append(HIST1[i][j])
#Returns the Jaccard Similarity Index value between lists A and B
def JacCompare(A,B):
	sum_min = 0
	A_row_sum = 0
	B_row_sum = 0
	for k in range(0, len(A)):
		sum_min += int(x[k] and y[k])
	A_row_sum = sum(x)
	B_row_sum = sum(y)
	if (A_row_sum == 0):
		A = 1
	if (B_row_sum == 0):
		B = 1
	similarity = sum_min/min(A_row_sum,B_row_sum)
	return similarity

def OptJacSim(col_mtrx):
	similarity = []
	#Build List
	for i in range(0, len(col_mtrx)):
		temp = []
		similarity.append(temp)
	for i in range(0, len(col_mtrx)):
		x = col_mtrx[i]
		for j in range(0, len(col_mtrx)):
			#Don't calculate inverse Similarities.
			if(j == i):
				similarity[i].append(1)
			elif(j >= i):
				y = col_mtrx[j]
				sum_min = 0
				A = 0
				B = 0
				for k in range(0, row_range):
					sum_min += int(x[k] and y[k])
				A = sum(x)
				B = sum(y)
				if (A == 0):
					A = 1
				if (B == 0):
					B = 1
				similarity[i].append(sum_min/min(A,B))
			else:
				similarity[i].append(similarity[j][i])
	return similarity
def JaccardSimilarity(col_mtrx):
	similarity = []
	for i in range(0, len(col_mtrx)):
		temp = []
		similarity.append(temp)
	for i in range(0, len(col_mtrx)):
		x = col_mtrx[i]
		for j in range(0, len(col_mtrx)):
			y = col_mtrx[j]
			sum_min = 0
			A = 0
			B = 0
			for k in range(0, row_range):
				sum_min += int(x[k] and y[k])
				A = sum(x)
				B = sum(y)
			if (A == 0):
				A = 1
			if (B == 0):
				B = 1
			similarity[i].append(sum_min/min(A,B))
	return similarity

def JaccardDifference(col_mtrx):
	diff_mtrx = []
	for i in range(0,len(col_mtrx)):
		temp = []
		diff_mtrx.append(temp)
		for j in range(0, len(col_mtrx[0])):
			diff_mtrx[i].append(1-col_mtrx[i][j])
	return diff_mtrx
#diff = JaccardDifference(sm)

#Removes rows of all 0s
def CleanData(mtrx):
	remove_idx = []
	idx = 0
	temp = mtrx
	for row in temp:
		if(sum(row) == 0):
			remove_idx.append(idx)
		idx += 1
	rem_end = len(remove_idx)
	for i in range(0, rem_end):
		idx = remove_idx.pop()
		temp.pop(idx)
	return temp

clean_data = CleanData(columns)


#Produce K arrays of data matching size of a row in col_mtrx
def RandData(col_mtrx, k):
	import random
	length = len(col_mtrx[0])
	data_list = []
	for i in range(0, k):
		data = []
		for j in range(0, length):
			data.append(random.randint(0,1))
		data_list.append(data)
	return data_list
#Calculate Jac. Dist. with random data added
def AppendRandData(rand_data, jacc_mtrx):
	temp = jacc_mtrx
	length = len(rand_data)
	for data in rand_data:
		temp.append(data)
	return [temp, length]
#diff = JaccardDifference(sm)
rdata = RandData(clean_data, 3)
rdist = AppendRandData(rdata, clean_data)
rdist_sm = OptJacSim(rdist[0])
rdist_diff = JaccardDifference(rdist_sm)
#sm_edit = list(map(list, zip(*sm)))
rdata_pts = rdist[1]

#Extract rand_data_pts number of rows from col_mtrx
def ExtractDiffMtrx(col_mtrx, rand_data_pts):
	temp = col_mtrx
	diff_mtrx = []
	for i in range(0, rand_data_pts):
		diff_mtrx.append(temp.pop())
	return diff_mtrx

#Contains only random data column comparisons
ext_diff = ExtractDiffMtrx(rdist_diff, rdata_pts)

#Takes a diff matrix and cluster labels and returns a list of matrices for each cluster
def CalcClusterMean(col_mtrx, cluster_array, cluster_pts):
	temp_diff = []
	#Build temp diff mtrx list
	for i in range(0, cluster_pts):
		temp = []
		temp_diff.append(temp)
	for i in range(0, len(col_mtrx)):
		#idx = cluster_array[i]
		#temp_diff[idx].append(col_mtrx[i])
		if(cluster_array[i] == 1):
			temp_diff[0].append(col_mtrx[i])
		elif(cluster_array[i] == 2):
			temp_diff[1].append(col_mtrx[i])
		elif(cluster_array[i] == 3):
			temp_diff[2].append(col_mtrx[i])
	return temp_diff

def MinAvgDistance(jac_mtrx):
	min_dist = 1000
	min_idx = 1000
	for i in range(0, len(jac_mtrx)):
		temp = sum(jac_mtrx[i])
		if(temp < min_dist):
			min_dist = temp
			min_idx = i
	return [min_dist, min_idx]

#Pick n random rows from jac_mtrx
def InitializeClusters(jac_mtrx, n):
	import random
	return [random.randint(0,len(jac_mtrx)) for i in range(0, n)]
#print(InitializeClusters(clean_data, 3))

#Take jaccard distance mtrx and cluster index. Assign each row of dist. mtrx to the cluster
#with minimum distance.
def AssignClusters(col_mtrx, clusters):
	new_clusters = []
	for i in range(0, len(col_mtrx)):
		min = 1000
		min_idx = 0
		for cluster_index in clusters:
			if(col_mtrx[i][cluster_index] < min):
				min = col_mtrx[i][cluster_index]
				min_idx = clusters.index(cluster_index)
		new_clusters.append(min_idx+1)
	return new_clusters

t_jac_sim = OptJacSim(clean_data)
t_jac_dist = JaccardDifference(t_jac_sim)
t_clusters = [0, 1, 2]
#InitializeClusters(t_jac_dist, 3)
#print(t_clusters)
#print(AssignClusters(t_jac_dist, t_clusters))

#Take a distance matrix and an array that labels the cluster of each row in the matrix
#For each cluster, find the minimum sum of the rows in the cluster. This is the new center.
#Return an array of indices for the new centers.
def FindMean(jac_mtrx, clusters, n_clusters):
	cluster_mins = []
	cluster_idx = []
	for i in range(0, n_clusters):
		cluster_mins.append(1000)
		cluster_idx.append(-1)
	for i in range(0, len(jac_mtrx)):
		cluster = clusters[i]
		row_sum = sum(jac_mtrx[i])
		if row_sum < cluster_mins[cluster-1]:
			cluster_mins[cluster-1] = row_sum
			cluster_idx[cluster-1] = i
	return cluster_idx

t_new_clusters = AssignClusters(t_jac_dist, t_clusters)
t_means = FindMean(t_jac_dist, t_new_clusters, 3)

#Take a diff. mtrx, original clusters index, and amount of times to iterate
#Return final array that labels each row in a cluster
def IterateFindMean(jac_mtrx, clusters, iterations):
	new_clusters = AssignClusters(jac_mtrx, clusters)
	cluster_amount = len(clusters)
	for i in range(0, iterations):
		new_means = FindMean(jac_mtrx, new_clusters, cluster_amount)
		new_clusters = AssignClusters(jac_mtrx, new_means)
		print(new_means)
#		print(new_clusters)
	return new_means

final = IterateFindMean(t_jac_dist, t_clusters, 3)
final_clusters = AssignClusters(t_jac_dist, final)
c = []
for i in range(0, 3):
	temp = []
	c.append(temp)
for i in range(0, len(final_clusters)):
	c_idx = final_clusters[i]-1
	c[c_idx].append(clean_data[i])

#Assess quality of clusters by finding the variance among each cluster
#Calculate std. dev. of cluster mean and calculates variance for each NP in cluster
#Takes a distance matrix, cluster center indices, and amount of clusters
#Returns an array where each element is that row's variance
def AssessQuality(jac_mtrx, clusters, n_clusters):
	new_clusters = AssignClusters(jac_mtrx, clusters)
	cluster_means = []
	cluster_diffs = []
	cluster_size = []
	variance = []
	#Get mean of clusters from center indices
	for i in range(0, n_clusters):
		temp = []
		cluster_means.append(sum(jac_mtrx[clusters[i]]))
		cluster_diffs.append(0)
		cluster_size.append(0)
	for i in range(0, len(jac_mtrx)):
		cluster_idx = new_clusters[i]-1
		cluster_means[cluster_idx] += sum(jac_mtrx[i])
		cluster_size[cluster_idx] += 1
	for i in range(0, n_clusters):
		cluster_means[i] = cluster_means[i]/cluster_size[i]
	print("cluster means: ", cluster_means)
	#For each row, calc its variance with its cluster
	for i in range(0, len(jac_mtrx)):
		cluster_idx = new_clusters[i]-1
		#square differences and sum them
		var = sum(jac_mtrx[i]) - cluster_means[cluster_idx]
		cluster_diffs[cluster_idx] += var**2
		#print("cluster_diffs ",cluster_idx, cluster_diffs[cluster_idx])
	for i in range(0,n_clusters):
		variance.append(cluster_diffs[i]/cluster_size[i])
	return variance

t_variance = AssessQuality(t_jac_dist, t_clusters, 3)
print("cluster variances: ", t_variance)

import matplotlib
import matplotlib.pyplot as plt

def JaccardPlot(jaccard_index, name):
	fig, ax = plt.subplots()
	im = ax.imshow(jaccard_index)
	ax.set_title(name)
	fig.tight_layout()
	plt.show()
i = 0
print(len(c[0]),len(c[1]),len(c[2]))
for x in c:
	i = i+1
	print(JaccardPlot(x, i))
#JaccardPlot(sm,"Jaccard Similarity Index")
#JaccardPlot(diff,"Jaccard Distance Index")
#JaccardPlot(rdist_sm,"Random Data Similarity Index")
#JaccardPlot(rdist_diff,"Random Data Distance Index")
#JaccardPlot(C1, "Cluster1")
#JaccardPlot(C2, "Cluster2")
#JaccardPlot(C3, "Cluster3")
