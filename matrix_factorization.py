import numpy as np
from scipy.sparse import lil_matrix
from scipy.stats import pearsonr as pearson
import math
import os
import pickle
import time
import matplotlib.pyplot as plt


#########################################
############### OPEN DATA ###############
#########################################

data_file = 'ml-100k/u.data'
data_file = open(data_file,'r')
users,movies = list(),list()

for line in data_file.readlines():
	line = line.split()
	user = int(line[0])
	movie = int(line[1])
	users.append(user)
	movies.append(movie)

users = np.unique(users)
movies = np.unique(movies)

user_to_id = dict(zip(users,range(len(users))))
movie_to_id = dict(zip(movies,range(len(movies))))
id_to_user = dict(zip(range(len(users)),users))
id_to_movie = dict(zip(range(len(movies)),movies))

data = lil_matrix((len(users),len(movies)))

data_file = 'ml-100k/u.data'
data_file = open(data_file,'r')
for line in data_file.readlines():
	line = line.split()
	user = user_to_id[int(line[0])]
	movie = movie_to_id[int(line[1])]
	rating = int(line[2])
	data[user,movie] = rating


#########################################
############# FACTORIZATION #############
#########################################

n = len(users)
m = len(movies)
K = 30
lr = .01
error_threshold = .5
split = .8

train_errors,test_errors = list(),list()

p = np.random.rand(n,K)
q = np.random.rand(m,K)

entries = list(zip(*data.nonzero()))
train_entries = entries[:int(len(entries)*split)]
test_entries = entries[int(len(entries)*split):]

def global_squared_error(r,p,q,local_entries):
	error = 0
	for (i,j) in local_entries:
		current_error = r[i,j]
		for k in range(K): current_error -= p[i,k]*q[j,k]
		error += abs(current_error)
	return error/len(local_entries)

global_error_train = global_squared_error(data,p,q,train_entries)
global_error_test = global_squared_error(data,p,q,test_entries)
iter=0
start = time.time()
while global_error_train > error_threshold and time.time()-start<120:
	print('\nMean absolute error at iteration',str(iter)+':') 
	print('\tTrain:',global_error_train)
	print('\tTest:',global_error_test)
	print('\tTime:',time.time()-start)
	iter+=1
	for (i,j) in train_entries: 
		local_error = data[i,j] - sum([p[i,k]*q[j,k] for k in range(K)])
		for k in range(K):
			p[i,k] += 2*lr*local_error*q[j,k]
			q[j,k] += 2*lr*local_error*p[i,k]
	global_error_train = global_squared_error(data,p,q,train_entries)
	global_error_test = global_squared_error(data,p,q,test_entries)


