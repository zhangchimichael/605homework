from pyspark import SparkContext
import sys
import math
import random
import numpy as np

#read args
F = int(sys.argv[1])
B = int(sys.argv[2])
iterations = int(sys.argv[3])
beta = float(sys.argv[4])
lamb = float(sys.argv[5])
inpath = sys.argv[6]
outWPath = sys.argv[7]
outHPath = sys.argv[8]

#initialize
tau = 100
sc = SparkContext("local")
#total v entries
totalV = 0

#v is B*B blocks. v[i][j] here will be list of all (i, j , i/B, j/B, rating) values
v = [[[] for i in xrange(B)] for j in xrange(B)]

#read input file
rows = 0
cols = 0
with open(inpath) as fin:
    for line in fin:
        splits = line.rstrip().split(',')
        x, y, r = int(splits[0]),int(splits[1]), float(splits[2])
        x-=1
        y-=1
        rows = max(rows, x)
        cols = max(cols, y)
        v[x%B][y%B].append((x, y, x/B, y/B, r))
        totalV +=1
        
rows+=1
cols+=1        
        
#block size for w blocks and h blocks        
xBlockSize = int(math.ceil(rows / float(B)))
yBlockSize = int(math.ceil(cols / float(B)))

#w and h are B blocks each. w[b] means the bth block which contains xBlockSize of vectors
w = [[np.array([random.random() for i in xrange(F)]) for k in xrange(xBlockSize)] for b in xrange(B)] 
h = [[np.array([random.random() for i in xrange(F)]) for k in xrange(yBlockSize)] for b in xrange(B)] 

#total numbers of entries for v rows and columns
Ni = [0 for i in xrange(rows)]
Nj = [0 for j in xrange(cols)]
for vI in v:
    for vIJ in vI:
        for i, j, i_B, j_B, r in vIJ:
            Ni[i]+=1
            Nj[j]+=1
            
#send to workers            
Ni = sc.broadcast(Ni)
Nj = sc.broadcast(Nj)
beta = sc.broadcast(beta)
lamb = sc.broadcast(lamb)
v = sc.broadcast(v)

#epsilon
def eps(n):
    global tau
    global beta
    return math.pow(tau+n, -beta.value)
  
    
#code for worker. for v block v(I, J), given corresponding w and h blocks, update w h and send back to master
def newWH(lists):
    global v
    global Ni
    global Nj
    global n
    global lamb
    n = n.value
    I, J, w, h = lists
    for i, j, i_B, j_B, r in v.value[I][J]:
        w_ = w[i_B]+2*eps(n)*((r - np.inner(w[i_B], h[j_B]))*h[j_B] - lamb.value/Ni.value[i]*w[i_B])
        h_ = h[j_B]+2*eps(n)*((r - np.inner(w[i_B], h[j_B]))*w[i_B] - lamb.value/Nj.value[j]*h[j_B])
        w[i_B] = w_
        h[j_B] = h_
        n+=1
    return (w, h)

    
#iList will remain steady. jList will shuffle every time    
iList = range(B)
jList = range(B)
for it in xrange(iterations):
    random.shuffle(jList)
    #ijList is a list with each entry (i, j, w[i], h[j])
    ijList = sc.parallelize(zip(iList, jList, map(lambda i:w[i], iList), map(lambda j:h[j], jList)), B)
    #update n
    n = sc.broadcast(it*totalV)
    i = 0
    #collect updates from each worker
    for newW, newH in ijList.map(lambda ij: newWH(ij)).collect():
        w[i] = newW
        h[jList[i]] = newH
        i+=1

#output. note ith row for w is stored as w[i%B][i/B]
with open(outWPath, 'w') as outW:
    for i in xrange(rows):
        print >> outW, ','.join([str(x) for x in w[i%B][i/B]])
        
with open(outHPath, 'w') as outH:
    for f in xrange(F):
        print >> outH, ','.join([str(h[j%B][j/B][f]) for j in xrange(cols)])
