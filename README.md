# 605 homework 7

how to run example

PATH=$PATH:/afs/cs.cmu.edu/project/bigML/spark-1.3.0-bin-hadoop2.4/bin
spark-submit ~/bigml/a7/dsgd_mf.py 20 3 1 0.9 0.1 autolab_train.csv w.csv h.csv

args:
F = int(sys.argv[1])
B = int(sys.argv[2])
iterations = int(sys.argv[3])
beta = float(sys.argv[4])
lamb = float(sys.argv[5])
inpath = sys.argv[6]
outWPath = sys.argv[7]
outHPath = sys.argv[8]

