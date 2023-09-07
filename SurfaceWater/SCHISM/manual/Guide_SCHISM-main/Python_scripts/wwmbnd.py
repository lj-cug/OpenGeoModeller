# @author: Krys1202
#Create wwmbnd.gr3 based on SELFE open bnd segments
import csv
import numpy as nu

#     Inputs:         
#     hgrid.gr3, 
#     gen_wwmbnd.in (see sample below)
#     Output: wwmbnd.gr3

#     Sample gen_wwmbnd.in
# 4 !ncond, i.e. number of conditions
# 1 2 2 !cond1 #seg flag ob
# 2 2 1 !cond2 #seg flag land
# 3 22 -1 !cond3 #seg flag island
# 4 99 0 !cond4 #dummy flag other nodes
	
#     Input files, info about flags for different boundaries
#     conditions ordered as they appear in hgrid.gr3
#     i.e. open, land, island
#     condition #, # boundaries, flag for open boundary nodes
#     condition #, # boundaries, flag for land boundary nodes
#     condition #, # boundaries, flag for island boundary nodes
#     condition #, # boundaries, flag for other nodes (set inside code as initial condition)

f = open('gen_wwmbnd.in')
bnd_in = f.readlines()
f.close()

ncond = int(bnd_in[0][0])
print('number of node condition flags is',ncond,' these are:')
for i in range(ncond):
    ii=i+1
    x = bnd_in[ii].split(' ')
    #print(x)
    print(f'condition {x[0]}, # boundaries {x[1]}, flag {x[2]}')

#read in details from hgrid
with open('hgrid.gr3') as f:
    fr = csv.reader(f,delimiter=' ')
    next(fr)
    nenp = next(fr)
    #initialise arrays (ne = number of elements; np = number of nodes)
    #xnd = lond, ynd = lat, ibnd = node number, nm = 3 nodes numbers per element 
    ne = int(nenp[0])
    np = int(nenp[1].strip())
    xnd =nu.zeros([np])
    ynd =nu.zeros([np])
    ibnd=nu.zeros([np], dtype='i')
    nm=nu.zeros([ne,3], dtype='i')

    print(ne,np,nm.shape)

    #read in xnd,ynd for each node
    for i in range(np):
        rec=next(fr)
        #print(rec)
        k=0
        for j in range(len(rec)):
            if rec[j] !='':
                if k == 1: xnd[i] = float(rec[j])
                if k == 2: ynd[i] = float(rec[j])
                k=k+1
    print(xnd[0],ynd[0])

    #read in node numbers for each element 
    for i in range(ne):
        rec = next(fr)
        if i==0: print(rec)
        for j in range(2,5):
            m = int(rec[j])
            nm[i,j-2] = m
            if i==0: print(rec,j,m,nm[i,j-2])

    print(nm[0,:],nm.dtype)


    rec = next(fr)
    nope = int(rec[0])
    print(nope)

    #check expected number of open boundaries
    x = bnd_in[1].split(' ')
    nope2 = int(x[1])
    if nope!=nope2: 
        print('nope/=nope2')
        exit()

    #ok, set up flag
    ifl_ob = int(x[2])

    neta = next(fr)

    for k in range(nope):
        rec=next(fr)
        nond = int(rec[0])
        print(nond)
        for i in range(nond):
            rec = next(fr)
            iond = int(rec[0])
            if i == 0: print(rec,iond)
            if iond>np or iond<=0:
                print('iond>np')
                exit()
            ibnd[iond-1]=ifl_ob

    #now land/island boundaries
    #total number of land/island boundaries
    rec = next(fr)
    nlandb = int(rec[0])
    #total number of land boundary nodes
    rec = next(fr)
    nlandn = int(rec[0])
    print('number of land/island boundaries',nlandb)
    print('number of land nodes',nlandn)

    #check expected number of land boundaries
    x2 = bnd_in[2].split(' ')
    x3 = bnd_in[3].split(' ')
    print(x2)
    print(x3)
    nlb = int(x2[1]) #number of land boundaries
    nib = int(x3[1]) #number of island boundaries
    nlib = nlb+nib
    if nlandb!=nlib: 
        print('nlandb/=nlb',nlandb,nlib)
        exit()

    ifl_l = int(x2[2])
    ifl_i = int(x3[2])
    print('flags for land and island',ifl_l,ifl_i)

    for k in range(nlandb):
        rec = next(fr)
        nond = int(rec[0])
        print(nond)
        for i in range(nond):
            rec = next(fr)
            iond = int(rec[0])
            if i == 0: print(rec,iond)
            if iond>np or iond<=0:
                print('iond>np')
                exit()
            #check if land or island boundary
            if k <= nlb-1:
                ibnd[iond-1]=ifl_l
            else:
                ibnd[iond-1]=ifl_i

#
#     Output
#
#     Output file
with open('wwmbnd.gr3','w') as w:
    ww = csv.writer(w,delimiter=' ')
    ww.writerow('')
    ww.writerow([ne,np])
    for i in range(np):
        ww.writerow([i+1,xnd[i],ynd[i],float(ibnd[i])])
    for i in range(ne):
        ww.writerow([i+1,3,nm[i,0],nm[i,1],nm[i,2]])
        
# check duplicated nodes
# check for the first node of open bnd
