import random
import numpy as np
import scanpy as sc




def getitem(data):
    rand_start = random.randint(0, data.shape[0]-1)
    full_seq = data[rand_start].toarray()[0]
    return full_seq

data = sc.read_h5ad('C:\\Users\\fengzhixin\\Documents\\scfoundation\\scfoundation\\fzx\\data\\Zheng68K.h5ad')
data = data.X

test = getitem(data)
print(test)
