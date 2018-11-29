import os
import numpy as np
import pandas as pd
import shutil
import array 


#open a file and parse it's data row by row
infile = open("input.txt", "r")
outfile=open("output.txt", "w")
n_col = 3
row = []
col_latency = [] 

for line in infile:
    row = line.split(':')
    col_latency.append(float(row[1]))
        

infile.close()
latency = np.array(col_latency)

#for i in range(len(latency)):
#    print("latency is: %f \n"% latency[i])

hist, bin_edges = np.histogram(latency, range=(0, latency.max()), bins = 100)
hist_normed, bin_edges_normed = np.histogram(latency, range=(0, latency.max()), bins = 100, normed=True)
mean = np.mean(latency)
std = np.std(latency)
dx=bin_edges_normed[1]-bin_edges_normed[0]
cdf= np.cumsum(hist_normed)*dx
for i in range(hist.size):
    outfile.write("%f \t %d\r\n" % (bin_edges[i]*1000000, hist[i]))

outfile.write("cdf latency: \n")
for i in range(hist.size):
    outfile.write("%f \t %f\r\n" % (bin_edges_normed[i]*1000000, cdf[i]))

outfile.write("mean latency \t %f\r\n" % (mean))
outfile.write("standard deviation in latency \t %f\r\n" % (std))

outfile.close()



