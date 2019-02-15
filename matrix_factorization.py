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
error_threshold = .1
split = .8

train_errors,test_errors = list(),list()

p = np.random.rand(n,K)
q = np.random.rand(m,K)
user_bias = np.random.rand(n)
movie_bias = np.random.rand(m)

entries = list(zip(*data.nonzero()))
n_train = int(len(entries)*split)
all_indices = np.arange(len(entries))
train_indices = np.random.choice(all_indices,n_train,replace=False)
train_entries = [entries[i] for i in train_indices]
test_indices = np.setdiff1d(all_indices,train_indices)
test_entries = [entries[i] for i in test_indices]
print(len(test_entries))


