#!/usr/bin/env python
""" 
GUI Tool for SCHISM netcdf out (schout_nc) visualization.!
Created on Tue Jan 29 14:52:24 2019
@author: JacobB

"""
__author__  = "Benjamin Jacob"
__license__ = "GNU GPL v2.0"
__email__ = "benjamin.jacob@hereon.de"

# speed up vecotr plotting
# use subaxes in figure

# polygon creation code
#fig2, ax2 = plt.subplots()
#fig2.show()
#selector2 = PolygonSelector(ax2, lambda *args: None)
#print("Click on the figure to create a polygon.")
#print("Press the 'esc' key to start a new polygon.")
#print("Try holding the 'shift' key to move all of the vertices.")
#print("Try holding the 'ctrl' key to move a single vertex.")


# salloc --x11 -p interactive -A gg0028 -n 10 -t 480

import sys
import glob
import dask
try:
	dask.config.set({"array.slicing.split_large_chunks": True})
except:
	pass	
import xarray as xr
import numpy as np
import scipy
from scipy.spatial import cKDTree
from tkinter import messagebox, ttk

# triy imprting cmpocean as prefered colormap
try: #use cmocean colormaps if installed else matplotlib 
	import cmocean.cm as cmo	
	use_cmocean=True
	print('using cmocean colormaps')                        
except:	#use classic matploib for color selection
	use_cmocean=False		
	print('using matplotlib colormaps')  
#use_cmocean=False	manually force classic colormaps
	
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
import datetime as dt
import matplotlib.backends.backend_tkagg as tkagg
from numpy import * # make available for eval expressions
import time


import warnings
warnings.filterwarnings("ignore")	

if sys.version_info> (3,0): # check tkinter python 2 / 3
    from tkinter import filedialog
    import tkinter as tk
else:
    import Tkinter as tk
    import tkFileDialog as filedialog
	
# put usage of cmocan in here but kind of an ugly implementation

# how to add comparing variables in debug mode
### add computational varname To list
#varname1='sedDepositionalFlux'
#varname2='sedErosionalFlux'
#varname='sedNetDeposition'
#self.varlist=list(self.varlist)+[varname,]
#self.vardict[varname]=varname
#self.ncs[varname]={varname:self.ncs[self.vardict[varname1]][varname1]-self.ncs[self.vardict[varname2]][varname2]}
#phi = 0 #porosity
# cumulated sedimient flux
#self.ncs[varname]={varname:((self.ncs[self.vardict[varname1]][varname1]-self.ncs[self.vardict[varname2]][varname2])*3600/(2650*1-phi)).cumsum()}
#self.varlist=list(np.sort(self.varlist))
#self.combo['values']=np.sort(self.combo['values']+(varname,))

#add diff mode

class param:		
	"""	functions for param.in for reading and editing. Operates in local directory """
	import os
	def __init__(self,fname='param.nml',comments='!'):
		#self.param_in=np.asarray(np.loadtxt(fname,comments=comments)[1:-1],int)-1		
		
		if '/' in fname:
			islash=fname.rindex('/')
			self.dir=fname[:islash]		
		else:
			self.dir='./'
			
		f=open(self.dir+'param.nml')	
		self.lines=f.readlines()
		f.close()
		
	def get_parameter(self,param='dt'):
		""" read parameter from param.in"""
		
		for line in self.lines:
			if param+' =' in line:
				param= line.split('=')[1].split('!')[0]
				try:
					param=float(param)
				except:
					param=str(param)
				break
		return param

	def set_parameter(self,params,values,outname='param.nml',outdir='./'):
		"""set_parameter(self,params,values,outname='param.nml',outdir='./') change parameters in param.in """
		if outname=='param.nml':
			try:
				os.rename('param.nml','param.nml.bkp')
			except:
				pass
		
		if type(params) == str:
			params=[params,]
			values=[values,]
		fout=open(outdir+outname,'w') 
		for line in self.lines:
			for param,value in zip(params,values):
				if param+' =' in line:
					line=' {:s} = {:.0f} !'.format(param,value)+line.split('!')[1]+'\n'
					values.remove(value)
					params.remove(param)
			fout.write(line)		

		fout.close()		
		print('updated param.nml has been loaded and will be accessed by get_parameters')	
		f=open(outdir+outname)	
		self.lines=f.readlines()
		f.close()	
	

class Window(tk.Frame):
    plt.ion() # notwendig  yes?    
		
    def __init__(self,master=None,ncdirsel=None,use_cmocean=use_cmocean):
        tk.Frame.__init__(self, master)               
        self.master = master
        self.use_cmocean=use_cmocean
        self.init_window(ncdirsel=ncdirsel)
                    
    def find_parent_tri(self,tris,xun,yun,xq,yq,dThresh=1000):
        """ parents,ndeweights=find_parent_tri(tris,xun,yun,xq,yq,dThresh=1000)
            find parent for coordinates xq,yq within triangulation tris,xun,yun.
            return: parent triangle ids and barycentric weights of triangle coordinates
        """    
        #% Distance threshold for Point distance
        dThresh=dThresh**2
        
        trisX,trisY=xun[tris],yun[tris]
        trinr=np.arange(tris.shape[0])
        
        #% orthogonal of side vecotrs
        SideX=np.diff(trisY[:,[0, 1, 2, 0]],axis=1)
        SideY=-np.diff(trisX[:,[0, 1, 2, 0]],axis=1)
        
        p=np.stack((xq,yq),axis=1)
        parent=-1*np.ones(len(p),int)
        for ip in range(len(p)):
                dx1=(p[ip,0]-trisX[:,0])
                dy1=(p[ip,1]-trisY[:,0])
                subind=(dx1*dx1+dy1*dy1) < dThresh # preselection
                subtris=trinr[subind]
                
                #% dot products
                parenti=(subtris[ (dx1[subind]*SideX[subind,0] + dy1[subind]*SideY[subind,0] <= 0) \
                               & ((p[ip,0]-trisX[subind,1])*SideX[subind,1] + (p[ip,1]-trisY[subind,1])*SideY[subind,1] <= 0) \
                                 & ( (p[ip,0]-trisX[subind,2])*SideX[subind,2] + (p[ip,1]-trisY[subind,2])*SideY[subind,2] <= 0) ][:])
                if len(parenti):
                    parent[ip]=parenti
        
        # tri nodes
        xabc=xun[tris[parent]]
        yabc=yun[tris[parent]]
        
        # barycentric weights
        divisor=(yabc[:,1]-yabc[:,2])*(xabc[:,0]-xabc[:,2])+(xabc[:,2]-xabc[:,1])*(yabc[:,0]-yabc[:,2])
        w1=((yabc[:,1]-yabc[:,2])*(xq-xabc[:,2])+(xabc[:,2]-xabc[:,1])*(yq-yabc[:,2]))/divisor
        w2=((yabc[:,2]-yabc[:,0])*(xq-xabc[:,2])+(xabc[:,0]-xabc[:,2])*(yq-yabc[:,2]))/divisor
        w3=1-w1-w2
        bttm=self.ncs[self.filetag][self.bindexname][0,:].values
        self.ibttms=np.asarray([(bttm[self.faces[parent[i],:]]).max()-1 for i in range(len(parent)) ],int)
        return parent,np.stack((w1,w2,w3)).transpose() 
		
    def vert_int(self,avg=False): 
        zcor=self.ncs[self.vardict[self.zcorname]][self.zcorname][self.total_time_index,:,:].values#self.ti_tk.get() #self.ncv['zcor']
        zcor=np.ma.masked_array(zcor,mask=self.mask3d)

		# scalar
        dz=np.diff(zcor,axis=-1)
        if not 'ivs' in self.ncs[self.vardict[self.varname]][self.varname].dims:
            data=self.ncs[self.vardict[self.varname]][self.varname][self.total_time_index,:,:].values
            dmean=0.5*(data[:,:-1]+data[:,1:])
            int=np.sum(dz*dmean,axis=-1)
			
            if avg:
                int/=(dz.sum(axis=-1))				
            return int		

        else:  #vector
            data=self.ncs[self.vardict[self.varname]][self.varname][:,self.total_time_index,:,:].values
            uabs=np.sqrt((data**2).sum(axis=0))
            
            dmean=0.5*(uabs[:,:-1]+uabs[:,1:])
            umean=0.5*(data[0,:,:-1]+data[0,:,1:])
            vmean=0.5*(data[1,:,:-1]+data[1,:,1:])
            int=np.sum(dz*dmean,axis=-1)
            intu=np.sum(dz*umean,axis=-1)
            intv=np.sum(dz*vmean,axis=-1)
			
            if avg:
                int/=(dz.sum(axis=-1))				
                intu/=(dz.sum(axis=-1))				
                intv/=(dz.sum(axis=-1))				
            return int, intu, intv		
			

    def schism_plotAtelems(self,nodevalues,add_cb=True):
        ph=plt.tripcolor(self.plotx,self.ploty,self.faces[:,:3],facecolors=self.nodevalues[self.faces[:,:3]].mean(axis=1),shading='flat',alpha=None) #test alpha = None for transparancy
        if add_cb:
            ch=plt.colorbar(extend='both')
        else:
            ch=None
        plt.tight_layout()
        return ph,ch
		
    def schism_updateAtelems(self):

        plt.figure(self.fig0)
        if self.quiver!=0:
            self.quiver.remove()
            self.arrowlabel.remove()   
            self.quiver=0	
        if self.CheckFixZ.get()!=0:                
            ibelow,iabove,weights=self.get_layer_weights(np.double(self.fixdepth.get()))
            lvl=str(self.fixdepth.get()+' m')
        else:
            lvl=self.lvl
        #from IPython import embed; embed()	          

		# timestep selection or average for on layer plots
        if self.shape[0]==self.nt:
            i0,i1=self.read_time_selection()
            self.ncvar=self.ncs[self.vardict[self.varname]][self.varname] #
            if self.CheckTavg.get():
                self.ncvar=self.ncvar[i0:i1+1,:].mean(dim='time')
            else:
                self.ncvar=self.ncvar[self.total_time_index,:]
        elif self.shape[1]==self.nt: #vector
            i0,i1=self.read_time_selection()		                        
            self.ncvar=self.ncs[self.vardict[self.varname]][self.varname] #
            if self.CheckTavg.get():
                self.ncvar=self.ncvar[:,i0:i1+1,:,:].mean(dim='time')
            else:
                self.ncvar=self.ncvar[:,self.total_time_index,:]

        elem_data=False # loaded data already at elements
        if (self.shape==(self.nt,self.nnodes,self.nz)) | (self.shape==(self.nts[0],self.nnodes,self.nz)):
            #self.ncvar=self.ncs[self.vardict[self.varname]][self.varname] # current variable
			#time average # weights for vert in will be wrong 
				
			# regular
            if self.CheckFixZ.get()==1:  # z interpolation
                #self.nodevalues=weights[0,:]*self.ncvar[self.total_time_index,:,:].values[self.nodeinds,ibelow]+weights[1,:]*self.ncvar[self.total_time_index,:,:].values[self.nodeinds,iabove]
                self.nodevalues=weights[0,:]*self.ncvar[:,:].values[self.nodeinds,ibelow]+weights[1,:]*self.ncvar[:,:].values[self.nodeinds,iabove]
				
				# optimize indexing
                self.nodevalues=np.ma.masked_array(self.nodevalues,mask=np.isnan(self.nodevalues))
            elif self.integrateZ.get()!=0: #z integrations
                self.nodevalues=self.vert_int(avg=False)
                lvl='z Int'
            elif self.avgZ.get()!=0: #z average	
                self.nodevalues=self.vert_int(avg=True)
                lvl='z avg'
            else: # regular surface slab
                #self.nodevalues=self.ncvar[self.total_time_index,:,self.lvl].values
                self.nodevalues=self.ncvar[:,self.lvl].values

				
            title=self.varname

        elif self.shape==(self.nnodes,):
            self.nodevalues=self.ncvar[:].values
            title=self.varname
            self.quiver=0
		# add at element variables variables
        elif (self.shape==(self.nt,self.nnodes)) | (self.shape==(self.nts[0],self.nnodes)):
            self.nodevalues=self.ncvar.values
            title=self.varname
        elif (self.shape==(self.nt,self.nelems)) | (self.shape==(self.nts[0],self.nnodes)):
            self.nodevalues=self.ncvar.values
            title=self.varname
            elem_data=True # loaded data already at elements
        elif (self.shape==(2,self.nt,self.nnodes)) | (self.shape==(2,self.nts[0],self.nnodes)): # 2 vector
            #u=self.ncvar[0,self.total_time_index,:].values
            #v=self.ncvar[1,self.total_time_index,:].values
            u=self.ncvar[0,:].values
            v=self.ncvar[1,:].values

            title='abs' + self.varname
            #self.nodevalues=np.sqrt(u*u+v*v)
            u=np.ma.masked_array(u,mask=self.drynodes)
            v=np.ma.masked_array(v,mask=self.drynodes)

			# need to differntiate for Tavg
            if self.CheckDiff.get()==1: #plot diffrence between absolute values					
                uabs1=np.sqrt((self.nclist[self.active_setup][self.vardict[self.varname]][self.varname][:,self.total_time_index,:]**2).sum(axis=0))
                uabs0=np.sqrt((self.nclist[0][self.vardict[self.varname]][self.varname][:,self.total_time_index,:]**2).sum(axis=0))
                self.nodevalues=(uabs1-uabs0).values
            else:
                if self.CheckTavg.get():			
                    self.nodevalues=np.sqrt((self.ncs[self.vardict[self.varname]][self.varname]**2).sum(axis=0)).values
                else:				
                    self.nodevalues=np.sqrt(u*u+v*v)

            #u=np.ma.masked_array(u,mask=np.isnan(u))
            #v=np.ma.masked_array(v,mask=np.isnan(v))
        elif (self.shape==(2,self.nt,self.nnodes,self.nz)) | (self.shape==(2,self.nts[0],self.nnodes,self.nz)):
            if self.CheckFixZ.get()==1: #vertical interpol (new code)
			
                #u=weights[0,:]*self.ncvar[0,self.total_time_index,:,:].values[self.nodeinds,ibelow]+weights[1,:]*self.ncvar[0,self.total_time_index,:,:].values[self.nodeinds,iabove]
                #v=weights[0,:]*self.ncvar[1,self.total_time_index,:,:].values[self.nodeinds,ibelow]+weights[1,:]*self.ncvar[1,self.total_time_index,:,:].values[self.nodeinds,iabove]
                u=weights[0,:]*self.ncvar[0,:,:].values[self.nodeinds,ibelow]+weights[1,:]*self.ncvar[0,:,:].values[self.nodeinds,iabove]
                v=weights[0,:]*self.ncvar[1,:,:].values[self.nodeinds,ibelow]+weights[1,:]*self.ncvar[1,:,:].values[self.nodeinds,iabove]
				
                if self.CheckDiff.get()==1: #plot diffrence between absolute values					
                    u0=weights[0,:]*self.nclist[0][self.vardict[self.varname]][self.varname][0,self.total_time_index,:,:].values[self.nodeinds,ibelow]+weights[1,:]*self.nclist[0][self.vardict[self.varname]][self.varname][0,self.total_time_index,:,:].values[self.nodeinds,iabove]
                    v0=weights[0,:]*self.nclist[0][self.vardict[self.varname]][self.varname][1,self.total_time_index,:,:].values[self.nodeinds,ibelow]+weights[1,:]*self.nclist[0][self.vardict[self.varname]][self.varname][1,self.total_time_index,:,:].values[self.nodeinds,iabove]
                    u1=weights[0,:]*self.nclist[self.active_setup][self.vardict[self.varname]][self.varname][0,self.total_time_index,:,:].values[self.nodeinds,ibelow]+weights[1,:]*self.nclist[self.active_setup][self.vardict[self.varname]][self.varname][0,self.total_time_index,:,:].values[self.nodeinds,iabove]
                    v1=weights[0,:]*self.nclist[self.active_setup][self.vardict[self.varname]][self.varname][1,self.total_time_index,:,:].values[self.nodeinds,ibelow]+weights[1,:]*self.nclist[self.active_setup][self.vardict[self.varname]][self.varname][1,self.total_time_index,:,:].values[self.nodeinds,iabove]
                    uabs1,uabs0=np.sqrt(u1**2+v1**2),np.sqrt(u0**2+v0**2)			

			# new io integrate			
            elif (self.integrateZ.get()!=0) | (self.avgZ.get()!=0): #z integrations	
                uabs,u,v=self.vert_int(avg=self.avgZ.get()!=0)
                if self.integrateZ.get()!=0:
                    lvl='z Int'
                else:	
                    lvl='z avg'						

                if self.CheckDiff.get()==1: #plot diffrence between absolute 
                    self.ncs=self.nclist[self.active_setup]
                    uabs1,u1,v1=self.vert_int(avg=self.avgZ.get()!=0)
                    self.ncs=self.nclist[0]
                    uabs0,u0,v0=self.vert_int(avg=self.avgZ.get()!=0)
                    self.ncs=self.diffnclist[self.active_setup-1]
                    u=u1-u0
                    v=v1-v0
						
            else: # regular surface slab
 
                #speed up for old io:
                if self.oldio:
                    hvel=self.ncs['schout']['hvel'][self.total_time_index,:,self.lvl,:]#.values
                    tmp=hvel.values
                    #u,v=hvel[:,0],hvel[:,1]
                    u,v=tmp[:,0],tmp[:,1]
                    if self.CheckDiff.get()==1: #plot diffrence between absolute values					
                        uabs1=np.sqrt(self.nclist[self.active_setup]['schout']['hvel'][self.total_time_index,:,self.lvl,:].sum(axis=0))
                        uabs0=np.sqrt(self.nclist[0]['schout']['hvel'][self.total_time_index,:,self.lvl,:].sum(axis=0))
						
                else: #new io
                    #u=self.ncvar[0,self.total_time_index,:,self.lvl].values
                    #v=self.ncvar[1,self.total_time_index,:,self.lvl].values
					# I preselected time for avraging possability
                    u=self.ncvar[0,:,self.lvl].values
                    v=self.ncvar[1,:,self.lvl].values
                    if self.CheckDiff.get()==1: #plot diffrence between absolute values					
                        uabs1=np.sqrt((self.nclist[self.active_setup][self.vardict[self.varname]][self.varname][:,self.total_time_index,:,self.lvl]**2).sum(axis=0))
                        uabs0=np.sqrt((self.nclist[0][self.vardict[self.varname]][self.varname][:,self.total_time_index,:,self.lvl]**2).sum(axis=0))					
					
				
            if self.CheckDiff.get()==0: #plot diffrence between absolute values
			
                if self.CheckTavg.get():			
                    print('using temporal avegarge of nodes {:s}-{:s} - not suitable if large variations in z'.format(self.exfrom.get(),self.exto.get()))                				
                    self.nodevalues=np.sqrt((self.ncs[self.vardict[self.varname]][self.varname][:,:,:,self.lvl]**2).sum(axis=0)).mean(axis=0).values
                else:				
                    self.nodevalues=np.sqrt(u*u+v*v)
                #self.nodevalues=np.sqrt(u*u+v*v)
            else:	
                if type(uabs1)==xr.core.dataarray.DataArray:
                    self.nodevalues=(uabs1-uabs0).values
                else:	
                    self.nodevalues=uabs1-uabs0
            u=np.ma.masked_array(u,mask=self.drynodes)
            v=np.ma.masked_array(v,mask=self.drynodes)
            #self.nodevalues=np.ma.masked_array(self.nodevalues,mask=np.isnan(self.nodevalues))
            #u=np.ma.masked_array(u,mask=np.isnan(u))
            #v=np.ma.masked_array(v,mask=np.isnan(v))
            title='abs ' + self.varname    
        else:
            print('variable shape missmatch - happened e.g. for unfinished runs with different outputwriting in variable files - assuming 3d file ')		
            if self.CheckFixZ.get()==0:
                self.nodevalues=self.ncvar[self.total_time_index,:,self.lvl].values
            else: # z interpolation
                self.nodevalues=weights[0,:]*self.ncvar[self.total_time_index,:,:].values[self.nodeinds,ibelow]+weights[1,:]*self.ncvar[self.total_time_index,:,:].values[self.nodeinds,iabove]
				# optimize indexing
                self.nodevalues=np.ma.masked_array(self.nodevalues,mask=np.isnan(self.nodevalues))
            title=self.varname
			
			
        if self.CheckEval.get()!=0: # evaluate on displayed variable
                expr= self.evalex.get()
                expr=expr[expr.index('=')+1:].replace('x','self.nodevalues').replace('A','self.A').replace('dt','self.dt')
                self.nodevalues=eval(expr)    
                
        if self.varname != 'depth':
            title=self.titlegen(lvl)
        else:
            title=self.varname
        # setting colorbar
        if elem_data:
            elemvalues=np.ma.masked_array(self.nodevalues[self.origins],mask=(self.dryelems==1)*self.maskdry.get())
        else:
            elemvalues=np.ma.masked_array(self.nodevalues[self.faces[:,:3]].mean(axis=1),mask=(self.dryelems==1)*self.maskdry.get())
		
        self.ph.set_array(elemvalues)
        if not self.CheckVar.get():
            cmin,cmax=np.nanmin(elemvalues),np.nanmax(elemvalues)			            
            self.clim=(cmin,cmax)			
            self.minfield.delete(0,tk.END)
            self.maxfield.delete(0,tk.END)
            self.minfield.insert(8,str(cmin))
            self.maxfield.insert(8,str(cmax))
        else:
            self.clim=(np.double(self.minfield.get()),np.double(self.maxfield.get()))
        self.ph.set_clim(self.clim) 
        
        # add quiver
        if self.quivVar.get():
            xlim=plt.xlim()
            ylim=plt.ylim()
            x=np.arange(xlim[0],xlim[1],(xlim[1]-xlim[0])/self.narrows)
            y=np.arange(ylim[0],ylim[1],(ylim[1]-ylim[0])/self.narrows)
            X, Y = np.meshgrid(x,y)
            if self.uselonlat.get()==0:
                d,qloc=self.xy_nn_tree.query((np.vstack([X.ravel(), Y.ravel()])).transpose()) #quiver locations
            else:
                d,qloc=self.ll_nn_tree.query((np.vstack([X.ravel(), Y.ravel()])).transpose()) #quiver locations				
            xref,yref=np.asarray(plt.axis())[[1,2]] +  np.diff(plt.axis())[[0,2]]*[- 0.2, 0.1]
            if (self.shape==(self.nt,self.nnodes)) or (self.shape==(self.nt,self.nnodes,self.nz)): 
                vmax=1.5#np.percentile(np.sqrt(u[qloc]**2+v[qloc]**2),0.95)
            else:
                vmax=np.double(self.maxfield.get())
            if self.shape[0]!=2:
                # load velocity for quiver plots on top of no velocity variables
                if self.oldio:
                    hvel=self.ncs['schout']['hvel'][self.total_time_index,:,self.lvl,:].values
                    u,v=hvel[:,0],hvel[:,1]
                else: #new io
                    #u=self.ncs[self.vardict[self.varname]][self.varname][0,self.total_time_index,:,self.lvl].values
                    #v=self.ncs[self.vardict[self.varname]][self.varname][1,self.total_time_index,:,self.lvl].values		
                    varname='horizontalVel'
                    u=self.ncs[self.vardict[varname]][varname][0,self.total_time_index,:,self.lvl].values
                    v=self.ncs[self.vardict[varname]][varname][1,self.total_time_index,:,self.lvl].values		
					

					
            u=np.ma.masked_array(u,mask=np.isnan(u))
            v=np.ma.masked_array(v,mask=np.isnan(v))			
            if self.normVar.get()==1:
                vabs=np.sqrt(u[qloc]*u[qloc]+v[qloc]*v[qloc])			
                #self.quiver=plt.quiver(np.concatenate((self.x[qloc],(xref,))),np.concatenate((self.y[qloc],(yref,))),np.concatenate((u[qloc]/vabs,(1,))),np.concatenate((v[qloc]/vabs,(0,))),scale=2,scale_units='inches') 
                self.quiver=plt.quiver(np.concatenate((self.plotx[qloc],(xref,))),np.concatenate((self.ploty[qloc],(yref,))),np.concatenate((u[qloc]/vabs,(1,))),np.concatenate((v[qloc]/vabs,(0,))),scale=2,scale_units='inches') 
                self.arrowlabel=plt.text(xref,yref,'normalized \n velocity ')
            else:				
                #self.quiver=plt.quiver(np.concatenate((self.x[qloc],(xref,))),np.concatenate((self.y[qloc],(yref,))),np.concatenate((u[qloc],(vmax,))),np.concatenate((v[qloc],(0,))),scale=2*vmax,scale_units='inches') 
                self.quiver=plt.quiver(np.concatenate((self.plotx[qloc],(xref,))),np.concatenate((self.ploty[qloc],(yref,))),np.concatenate((u[qloc],(vmax,))),np.concatenate((v[qloc],(0,))),scale=2*vmax,scale_units='inches') 
                self.arrowlabel=plt.text(xref,yref,'\n'*3+str(np.round(vmax,4))+' m/s')
 
		# multifig
        for fignr in self.plot_windows.keys():
            anno=self.plot_windows[fignr]['anno']
            if fignr not in plt.get_fignums() and (len(anno)>0) :
                for item in anno:
                    item.remove()
                self.plot_windows[fignr]['anno']=[]
        
        #update plot    
        plt.title(title)
        self.update_plots()
        print("done plotting ")

    def update_plots(self):
        for figi in plt.get_fignums()[1:]:
            try:
                plt.clim(self.clim) # not sure    
            except:
                #print('error updating clim')			                
                pass
				
	#def trigger_extractions(self):			
    #    nr=self.fignums
    #    self.plot_windows[self.fignums]={'fh':fig2,'nr':nr,'anno':anno,'type':self.timeseries,'coords':self.coords.copy(),'extract':self.extract,'xs':self.xs.copy()}
    #    self.plot_windows[self.fignums]['p0']=self.pt0i
    #    self.plot_windows[self.fignums]['varname']=self.varname
    #    self.plot_windows[self.fignums]['setup']=self.active_setup				
    #
    #    for fignr in self.plot_windows.keys():
    #        extract=self.plot_windows[fignr]['extract']
	#		
	#		# overwrite transect plots
    #        if fignr in plt.get_fignums() and (extract==self.profiles or extract==self.transect_callback):
    #            self.activefig=self.plot_windows[fignr]['nr']
    #            self.xs=self.plot_windows[fignr]['xs'].copy()
    #            self.coords=self.plot_windows[fignr]['coords'].copy()
    #            extract(self.plot_windows[fignr]['coords'])
    #        else:        
    #            self.plot_windows.pop(fignr)			
			
			        
    def titlegen(self,lvl):
        prefix=['',' $\Delta$ '][ self.CheckDiff.get()]	
        i0,i1=self.read_time_selection()
        prefix+=['',' <' + str(self.dates[i0])[:12]+' - '+str(self.dates[i1])[:12]+'>\n'][ self.CheckTavg.get()]	
		
        if self.oldio:
                return prefix+self.varname+' @ ' + str(self.ncs[self.filetag]['time'][self.total_time_index].values)[:16] + ' level= ' + str(lvl)
        elif self.newio<2: # new io
                return prefix+self.varname+' @ ' + str(self.reftime + dt.timedelta(seconds=int(self.ncs[self.filetag]['time'][self.total_time_index]))) + ' level= ' + str(lvl)		
        else:
                return prefix+self.varname+' @ ' + str(self.ncs[self.filetag]['time'][self.total_time_index].values)[:19] + ' level= ' + str(lvl)		

		
    def load_setup_data(self,ncdirsel=None):
        # load files    
        if ncdirsel==None:
            print("navigate into schsim run directory (containing param.nml)")
            self.runDir=filedialog.askdirectory(title='enter run directory direcory')+'/'
            nnodes=int(np.loadtxt(self.runDir+'hgrid.ll',skiprows=1,max_rows=1)[1])
            m=np.loadtxt(self.runDir+'hgrid.ll',skiprows=2,max_rows=nnodes)
            self.lon,self.lat=m[:,1],m[:,2]
            self.ll_nn_tree = cKDTree([[self.lon[i],self.lat[i]] for i in range(len(self.lon))])
            print("navigate into schout_*.nc directory")
            self.combinedDir=filedialog.askdirectory(title='enter schout_*.nc direcory')+'/'
        else:
            self.combinedDir=ncdirsel
			
		# new i/o files	
        if len(glob.glob(self.combinedDir+'out2d_*.nc'))>0:
            print('found per variable netcdf output format')
            self.oldio=False
            self.newio=1
        elif len(self.combinedDir+'schout_*.nc')>0:
            print('found schout.nc output format')
            self.oldio=True
            self.newio=0

		
        self.files=[] 		
        if self.oldio:
            for iorder in range(6): # check for schout_nc files until 99999
                self.files+=glob.glob(self.combinedDir+'schout_'+'?'*iorder+'.nc')
            self.stack_nrs=[int(file[file.rfind('_')+1:file.index('.nc')]) for file in self.files]
            self.files=list(np.asarray(self.files)[np.argsort(self.stack_nrs)])
            self.stack_nrs=list(np.asarray(self.stack_nrs)[np.argsort(self.stack_nrs)])
            self.nstacks=len(self.files)
            print('found ' +str(self.nstacks) +' stack(s)')
            self.stack0=int(self.stack_nrs[0])
            
            # initialize extracion along files # check ofr future if better performance wih xarray
            ncs={'schout':[]}
            ncs['schout']=xr.concat([ xr.open_dataset(self.combinedDir+'schout_'+str(nr)+'.nc').chunk() for nr in self.stack_nrs],dim='time')
            self.ncv=ncs['schout'].variables
            try:
                ncs['schout']=xr.concat([ xr.open_dataset(self.combinedDir+'schout_'+str(nr)+'.nc').chunk() for nr in self.stack_nrs],dim='time')
            except:
                print("error loading via MFDataset - time series and hovmoeller diagrams wont work")
                pass		
				
				
            self.vardict={} # variable to nc dict relations
			
           # load variable list from netcdf #########################################            
            exclude=['time','SCHISM_hgrid', 'SCHISM_hgrid_face_nodes', 'SCHISM_hgrid_edge_nodes', 'SCHISM_hgrid_node_x',
         'SCHISM_hgrid_node_y', 'bottom_index_node', 'SCHISM_hgrid_face_x', 'SCHISM_hgrid_face_y', 
         'ele_bottom_index', 'SCHISM_hgrid_edge_x', 'SCHISM_hgrid_edge_y', 'edge_bottom_index',
         'sigma', 'dry_value_flag', 'coordinate_system_flag', 'minimum_depth', 'sigma_h_c', 'sigma_theta_b', 
         'sigma_theta_f', 'sigma_maxdepth', 'Cs', 'wetdry_node','wetdry_elem', 'wetdry_side'] # exclude for plot selection
            vector_vars=[] # stack components for convenience	  
            self.vardict={} # variable to nc dict relations

            for vari in self.ncv:
                if vari not in exclude:
                    if  self.ncv[vari].shape[-1]==2:
                        vector_vars.append(vari)		
                        self.vardict[vari]=vari			
                        ncs[vari] ={vari: xr.concat([ncs['schout'][vari].sel(two=0),ncs['schout'][vari].sel(two=1)], dim='ivs')}
                    else:
                        self.vardict[vari]='schout'		
            self.varlist=list(self.vardict.keys())
            self.filetag='schout'			
            self.bindexname='node_bottom_index'
            self.zcorname='zcor'			
            self.dryvarname='wetdry_elem'
            if 'wetdry_node' not in self.ncv.keys():
                print('no wet dry node in keys')
                self.drynodename=False
            else:	
                self.drynodename='wetdry_node'
            self.hvelname='hvel'
            self.vertvelname='vertical_velocity'			
			# work around to map old velocity as new velocity formatted

            strdte=[np.float(digit) for digit in ncs[self.filetag]['time'].attrs['base_date'].split()]
            self.reftime=dt.datetime.strptime('{:04.0f}-{:02.0f}-{:02.0f} {:02.0f}:{:02.0f}:{:02.0f}'.format(strdte[0],strdte[1],strdte[2],strdte[3],strdte[3],0),'%Y-%m-%d %H:%M:%S')
				
            self.nclist.append(ncs)
            self.w0.config(to=len(self.nclist)-1)
			
            if len(self.nclist)>1:
                ncdiff={}
                for key in  self.nclist[0].keys():
                    if type(self.nclist[1][key])==dict:
                        if ('zcoor' in key.lower()) or ('zcor' in key.lower()):
                            ncdiff[key]=self.nclist[1][key][key]						
                        else:						
                            ncdiff[key]=self.nclist[1][key][key]-self.nclist[0][key][key]
                    else:
                        ncdiff[key]=self.nclist[1][key]-self.nclist[0][key]
                self.diffnclist.append(ncdiff)
                self.nstacks=self.nstacks0
            else:
                self.nstacks0=self.nstacks			
			
        else: # new io
            self.hvelname='horizontalVel'
            self.filetag='out2d'
            self.bindexname='bottom_index_node'
            self.zcorname='zCoordinates'
            self.dryvarname='dryFlagElement'
            self.drynodename='dryFlagNode'			
            self.vertvelname='verticalVelocity'			
            for iorder in range(8): # check for schout_nc files until 99999
                self.files+=glob.glob(self.combinedDir+'out2d_'+'?'*iorder+'.nc')
            self.stack_nrs=[int(file[file.rfind('_')+1:file.index('.nc')]) for file in self.files]
            self.files=list(np.asarray(self.files)[np.argsort(self.stack_nrs)])
            self.stack_nrs=list(np.asarray(self.stack_nrs)[np.argsort(self.stack_nrs)])
            self.nstacks=len(self.files)
            self.stack0=int(self.stack_nrs[0])
            print('found ' +str(self.nstacks) +' stack(s)')
    
            # file access        
    		# vars # problem sediment
            varfiles=[file[file.rindex('/')+1:file.rindex('_')] for file in glob.glob(self.combinedDir+'*_'+str(self.stack_nrs[0])+'.nc') ]
            ncs=dict.fromkeys(varfiles)
            try:
                for var in varfiles:
                    ncs[var]=xr.concat([ xr.open_dataset(self.combinedDir+var+'_'+str(nr)+'.nc').chunk() for nr in self.stack_nrs],dim='time')
            except:	
                self.stack_nrs=list(np.asarray(self.stack_nrs)[np.argsort(self.stack_nrs)])[:-1]
                self.nstacks=len(self.files)-1
                for var in varfiles:
                    ncs[var]=xr.concat([ xr.open_dataset(self.combinedDir+var+'_'+str(nr)+'.nc').chunk() for nr in self.stack_nrs],dim='time')		
		

           # load variable list from netcdf #########################################            
            exclude=['time','SCHISM_hgrid', 'SCHISM_hgrid_face_nodes', 'SCHISM_hgrid_edge_nodes', 'SCHISM_hgrid_node_x',
         'SCHISM_hgrid_node_y', 'bottom_index_node', 'SCHISM_hgrid_face_x', 'SCHISM_hgrid_face_y', 
         'ele_bottom_index', 'SCHISM_hgrid_edge_x', 'SCHISM_hgrid_edge_y', 'edge_bottom_index',
         'sigma', 'dry_value_flag', 'coordinate_system_flag', 'minimum_depth', 'sigma_h_c', 'sigma_theta_b', 
         'sigma_theta_f', 'sigma_maxdepth', 'Cs', 'dryFlagElement', 'dryFlagSide'] # exclude for plot selection
            vector_vars=[] # stack components for convenience	  
            self.vardict={} # variable to nc dict relations
            for nci_key in ncs.keys():
                for vari in ncs[nci_key].keys():
                    if vari not in exclude:
                        self.vardict[vari]=nci_key	
                    if vari[-1] =='Y': 
                        vector_vars.append(vari[:-1])
    	
            self.varlist=list(self.vardict.keys())
    
            for vari_vec in vector_vars:			
                varX=vari_vec+'X'	  
                varY=vari_vec+'Y'	  
                self.varlist+=[vari_vec]
                self.vardict[vari_vec]=vari_vec
                ncs[vari_vec] ={vari_vec: xr.concat([ncs[self.vardict[varX]][varX], ncs[self.vardict[varY]][varY]], dim='ivs')}

            p=param(self.runDir+'/param.nml')
            if self.newio==1:
                self.reftime=dt.datetime(int(p.get_parameter('start_year')),
                int(p.get_parameter('start_month')),
                int(p.get_parameter('start_day')),
                int(p.get_parameter('start_hour')),0,0)		

            self.nclist.append(ncs) # add ncs to list
            self.w0.config(to=len(self.nclist)-1)
			
        self.nts.append(len(ncs[self.filetag]['time']))
        isetup=len(self.nts)-1
		#
        if len(self.nclist)>1:
            ncdiff={}
            for key in  self.nclist[0].keys():
                if type(self.nclist[isetup][key])==dict:
                    ncdiff[key]={key:self.nclist[isetup][key][key]-self.nclist[0][key][key]} #check					
                else:
                    if ('zcoor' in key.lower()) or ('zcor' in key.lower()):
                        ncdiff[key]=self.nclist[isetup][key]
                    else:						
                        ncdiff[key]=self.nclist[isetup][key]-self.nclist[0][key]
			# reset dryelems
            ntmin=np.min((self.nts[0],self.nts[len(self.nclist)-1]))
            ncdiff[self.filetag][self.dryvarname]=np.maximum(self.nclist[1][self.filetag][self.dryvarname][:ntmin,:],self.nclist[0][self.filetag][self.dryvarname][:ntmin,:])				
						
            self.diffnclist.append(ncdiff)
            self.nstacks=self.nstacks0
        else:
            self.nstacks0=self.nstacks

    def init_window(self,ncdirsel=None):
        self.pack(fill=tk.BOTH,expand=1)
        self.fig0=len(plt.get_fignums())+1
        self.plot_windows={}  
        self.fignums=1
        self.figs=[]    #self.figs=[]  plt.get_fignums()
        self.annos=[]
        self.pt0=0      # coutner for annotation enumerator    
		
        # load grid and netcf file access for setup parotally used later	
        self.nclist=[]
        self.diffnclist=[]
        self.nts=[] # number of time steps for different setups
		# mv setup gui variable here to set variable beore first call
        self.stp_tk = tk.IntVar(self,value=0)
        self.w0=tk.Spinbox(self, from_=0,to=0,width=5,validate='all',textvariable=self.stp_tk) # try command and
        self.load_setup_data(ncdirsel)
        self.active_setup=0
        self.ncs=self.nclist[0]		
        #if self.newio<2:
        try:
             self.dates=np.asarray([self.reftime + dt.timedelta(seconds=ti) for ti in            self.ncs[self.filetag]['time'].values],np.datetime64)	            
        except:
             self.newio=2
             self.dates=self.ncs[self.filetag]['time'].values            		
        self.dt=self.dates[1]-self.dates[0]
        #self.reftime=dt.datetime.strptime(self.nc['time'].units[14:33],'%Y-%m-%d %H:%M:%S')
        self.varlist=list(np.sort(self.varlist))		
        self.nt,self.nnodes,self.nz,=self.ncs[self.vardict[self.zcorname]][self.zcorname].shape
        self.nodeinds=range(self.nnodes)
        
        if self.nt == 1:
            self.faces=np.asarray(self.ncs[self.filetag]['SCHISM_hgrid_face_nodes'][:].values-1,int)
            self.x=self.ncs[self.filetag]['SCHISM_hgrid_node_x'][:].values
            self.y=self.ncs[self.filetag]['SCHISM_hgrid_node_y'][:].values
        else: # file dimension become cocantenated by files
            self.faces=np.asarray(self.ncs[self.filetag]['SCHISM_hgrid_face_nodes'][0,:].values-1,int)
            try:
                self.x=self.ncs[self.filetag]['SCHISM_hgrid_node_x'][0,:].values
                self.y=self.ncs[self.filetag]['SCHISM_hgrid_node_y'][0,:].values
            except:
                self.x=self.ncs[self.filetag]['SCHISM_hgrid_node_x'][:].values
                self.y=self.ncs[self.filetag]['SCHISM_hgrid_node_y'][:].values
			
        try:
            self.stack_size=self.ncs[self.filetag].chunksizes['time'][0]
        except:
            self.stack_size=self.ncs[self.filetag].chunks['time'][0]
			
        try:
            lmin = self.ncs[self.filetag][self.bindexname][un0,:].values
        except:
            zvar = self.ncs[self.vardict[self.zcorname]][self.zcorname][0]
            lmin = np.zeros(self.x.shape,dtype='int')
            for i in range(len(self.x)):
                try:
                    lmin[i] = max(np.where(zvar.mask[i])[0])
                except:
                    lmin[i] = 0
                    lmin = lmin+1
        self.ibbtm=self.ncs[self.filetag][self.bindexname][0,:].values-1    
		
        self.mask3d=np.zeros((self.nnodes,self.nz),bool) # mask for 3d field at one time step
        for inode in range(self.nnodes):
            self.mask3d[inode,:self.ibbtm[inode]]=True # controlled that corresponding z is depths		
        self.mask_hvel=np.tile(self.mask3d,(2,1,1))
        self.mask_wind=self.mask_hvel[:,:,0]
        # next neighbour node look up tree
        self.xy_nn_tree = cKDTree([[self.x[i],self.y[i]] for i in range(len(self.x))]) # cart
        self.ll_nn_tree = cKDTree([[self.lon[i],self.lat[i]] for i in range(len(self.x))])# lonlat
        
        self.minavgdist=np.min(np.sqrt(np.abs(np.diff(self.x[self.faces[:,[0,1,2,0]]],axis=1))**2+np.abs(np.diff(self.y[self.faces[:,[0,1,2,0]]],axis=1))**2).mean(axis=1))

        # mesh for mesh visualisazion        
        xy=np.c_[self.x,self.y]
        self.mesh_tris=xy[self.faces[np.where(self.faces[:,-1]<0)][:,:3]]#[:,:3]
        self.mesh_quads=xy[self.faces[np.where(self.faces[:,-1]>-1)]][:,:]
		
        self.tripc = PolyCollection(self.mesh_tris,facecolors='none',edgecolors='k',linewidth=0.2) #, **kwargs)
        self.hasquads = np.min(self.mesh_quads.shape)>0
        
        if self.hasquads: #build tri only grid for faster plotting
            self.quadpc = PolyCollection(self.mesh_quads,facecolors='none',edgecolors='r',linewidth=0.2) #, **kwargs)    
            print("building pure triangle grid for easier plotting")
            faces2=[]
            self.origins=[] # mapping from nodes to triangles for triplot

            for nr,elem in enumerate(self.faces):
                if elem[-1]<0:
                    faces2.append(elem[:3])
                    self.origins.append(nr)
                else: # split quad into tris
                    faces2.append(elem[[0,1,2]])
                    faces2.append(elem[[0,2,3]])
                    self.origins.append(nr)
                    self.origins.append(nr)
            self.faces=np.array(faces2)                    
            self.origins=np.array(self.origins)
            self.nelems=self.origins.max()+1
            print("done")
        else:
             self.faces=self.faces[:,:3]
             self.origins=np.arange(self.faces.shape[0])
        ##########################################  
        self.dryelems=self.ncs[self.filetag][self.dryvarname][0,:][self.origins]
        if (self.drynodename == False) and (self.oldio==True):
            self.drynodes=self.ncs[self.vardict['hvel']]['hvel'][0,0:,-1].values==0
        else:
            self.drynodes=np.asarray(self.ncs[self.filetag][self.drynodename][0,:],bool)		

        # next neighbour element look up tree
        self.cx,self.cy=np.mean(self.x[self.faces],axis=1),np.mean(self.y[self.faces],axis=1)
        elcoords=[[self.cx[i],self.cy[i]] for i in range(len(self.cx))] # pooint pairs of nodes
        self.elem_nn_tree = cKDTree(elcoords) # next neighbour search tree	      

        ########################################################################## 
        #### GUI ELEMENTS ##############
        menubar = tk.Menu(self.master)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Exit", command=self.client_exit)
        menubar.add_cascade(label="File", menu=filemenu)
        saveMenu = tk.Menu(self.master)

        filemenu.add_cascade(label="Save", menu=saveMenu)
        saveMenu.add_command(label="CurrentNodeData", command=self.savenodevalues)
        saveMenu.add_command(label="Coordinates", command=self.savecoords)
        saveMenu.add_command(label="Timeseries", command=self.savets)
        saveMenu.add_command(label="Hovmoeller", command=self.savehov)
        saveMenu.add_command(label="Profiles", command=self.profiles)
        saveMenu.add_command(label="Transect", command=self.transect_callback)

        extractmenu = tk.Menu(menubar, tearoff=0)
        extractmenu.add_command(label="Timeseries", command=self.timeseries)
        extractmenu.add_command(label="Hovmoeller", command=self.hovmoeller)
        extractmenu.add_command(label="Horiz Hovmoeller", command=self.hovmoeller_horz)
        extractmenu.add_command(label="Profiles", command=self.profiles)
        extractmenu.add_command(label="Transect", command=self.transect_callback)
        extractmenu.add_command(label="Transect_from_bp_file", command=self.bptransect_callback)
        menubar.add_cascade(label="Extract", menu=extractmenu)
        self.master.config(menu=menubar)

		# add setup sel
        row=0
        L1=tk.Label(self,text='setup:')
        L1.grid(row=row,column=0)#.place(x=260,y=0)
		
		# variable declaration moved as workadournd
        #self.stp_tk = tk.IntVar(self,value=0)
        #self.w0=tk.Spinbox(self, from_=0,to=len(self.nclist)-1,width=5,validate='all',textvariable=self.stp_tk) # try command and
        self.w0.grid(row=row,column=1)
        self.stp_tk.trace("w",self.stp_callback) # activate call back after set        
		
        add_setup = tk.Button(self,text='add setup',command=self.load_setup_data)
        add_setup.grid(row=row,column=2)
		
        self.CheckDiff = tk.IntVar(value=0) # move before  plotAtElems definition to avoid error
        checkdiff=tk.Checkbutton(self,text='show diff',variable=self.CheckDiff,command=self.stp_callback)#,command=self.CheckDiff_callback)
        checkdiff.grid(sticky = tk.W,row=row,column=3)
		
        row=1
        L1=tk.Label(self,text='variable')
        L1.grid(row=row,column=0)#.place(x=260,y=0)

        L2=tk.Label(self,text='layer')
        L2.grid(row=row,column=1)#.place(x=200,y=0)

        L3=tk.Label(self,text='stack')
        L3.grid(row=row,column=2)

        L2=tk.Label(self,text='timestep')
        L2.grid(row=row,column=3)#.place(x=140,y=0)

        row+=1
        self.variable = tk.StringVar(self)
        self.variable.set('depth') # default value
        self.variable.trace("w",self.variable_callback)
        #w3=(self, self.variable, *self.varlist) #w3.config(width=10)
        #w3.grid(row=row,column=0)

        #in case to long use treeview		
        #w3=(self, self.variable, *self.varlist) #w3.config(width=10)
        #combo = ttk.Combobox(self,values=self.varlist)
        self.combo = ttk.Combobox(self,values=self.varlist)
        self.combo.bind("<<ComboboxSelected>>", self.variable_selection)
        self.combo.set('depth')
        #combo.place(x=50, y=50)
        self.combo.grid(row=row,column=0)	
		
        levels=[int(k) for k in range(self.ncs[self.vardict[self.zcorname]][self.zcorname].shape[2])] #spinwheel
        self.lvl_tk = tk.IntVar(self,value=str(np.max(levels)))
        self.quiver=0
        self.lvl=levels[-1]   # spinwheel
        w4=tk.Spinbox(self, from_=0,to=levels[-1],width=5,validate='all',textvariable=self.lvl_tk) # try command and
        w4.grid(row=row,column=1)
        self.lvl_tk.trace("w",self.lvl_callback) # activate call back after set

        self.stacks=self.stack_nrs
        self.stack = tk.IntVar(self)
        self.stack.set(self.stack_nrs[0]) 
        self.stack.trace("w",self.stack_callback)
        self.current_stack=self.stack_nrs[0]
        w=tk.Spinbox(self, values=self.stack_nrs, width=5,validate='all',textvariable=self.stack) # try command and set values in function with checking
        w.grid(row=row,column=2)

        self.ti_tk = tk.IntVar(self)
        self.ti=0
        self.ti_tk.trace("w",self.ti_callback)
        w2=tk.OptionMenu(self, self.ti_tk, *range(self.stack_size))#)#range(len(self.ncs['out2d']['time'])))
        w2.grid(row=row,column=3)

        row+=1
        start = tk.Button(self,text='|<<',command=self.firstts)
        start.grid(row=row,column=0)
        play = tk.Button(self,text='play',command=self.playcallback)
        play.grid(row=row,column=1)
        self.play=True
        stop = tk.Button(self,text='stop',command=self.stopcallback)
        stop.grid(row=row,column=2)
        end = tk.Button(self,text='>>|',command=self.lastts)
        end.grid(row=row,column=3)

        row+=1 # z  lvel interpolation
        self.CheckFixZ = tk.IntVar(value=0) # move before  plotAtElems definition to avoid error
        fixz=tk.Checkbutton(self,text='interp to z(m):',variable=self.CheckFixZ)
        fixz.grid(sticky = tk.W,row=row,column=0)
        self.fixdepth=tk.Entry(self,width=8)
        self.fixdepth.grid(row=row,column=1)

        self.integrateZ = tk.IntVar(value=0) # move before  plotAtElems definition to avoid error
        intz=tk.Checkbutton(self,text='z int',variable=self.integrateZ)
        intz.grid(sticky = tk.W,row=row,column=2)

        self.avgZ = tk.IntVar(value=0) # move before  plotAtElems definition to avoid error
        avgz=tk.Checkbutton(self,text='z avg',variable=self.avgZ)
        avgz.grid(sticky = tk.W,row=row,column=3)

		
        row+=1 # eval results
        self.CheckEval = tk.IntVar(value=0)
        fixz=tk.Checkbutton(self,text='eval:',variable=self.CheckEval)
        fixz.grid(sticky = tk.W,row=row,column=0)
        self.evalex=tk.Entry(self,width=16)
        self.evalex.grid(row=row,column=1,columnspan=2)
        self.evalex.insert(8,'x=x')
		
        self.CheckTavg = tk.IntVar(value=0)
        fixT=tk.Checkbutton(self,text='avg time',variable=self.CheckTavg)#,command=self.Tavg_callback)
        fixT.grid(sticky = tk.W,row=row,column=3)
        
        row+=1 # Apperance
        h1=tk.Label(self,text='\n Appearance:',anchor='w',font='Helvetica 10 bold')
        h1.grid(row=row,column=0)

        row+=1 # Clorors 
        self.CheckVar = tk.IntVar(value=0)
        fixbox=tk.Checkbutton(self,text='fix colorbar',variable=self.CheckVar)
        fixbox.grid(sticky = tk.W,row=row,column=0)

        # colormap
        self.cmap = tk.StringVar(self)
        self.cmap.trace("w",self.cmap_callback)
        if self.use_cmocean:
            maps=cmo.cmapnames		
            #self.cmap.set('haline') # default value						
        else:
            maps=np.sort([m for m in plt.cm.datad if not m.endswith("_r")])
            #self.cmap.set('jet') # default value			

        w4=tk.OptionMenu(self, self.cmap, *maps)
        w4.grid(row=row,column=1)

        row+=1
        l4=tk.Label(self,text='caxis:')
        l4.grid(row=row+1,column=0)
        l5=tk.Label(self,text='min:')
        l5.grid(row=row,column=1)
        self.minfield=tk.Entry(self,width=8)
        self.minfield.grid(row=row+1,column=1)
        l6=tk.Label(self,text='max:')
        l6.grid(row=row,column=2)
        self.maxfield=tk.Entry(self,width=8)
        self.maxfield.grid(row=row+1,column=2)

        row+=2
        self.xminvar = tk.StringVar()
        self.xminvar.trace("w",self.updateaxlim)
        self.xmaxvar = tk.StringVar()
        self.xmaxvar.trace("w",self.updateaxlim)
        self.yminvar = tk.StringVar()
        self.yminvar.trace("w",self.updateaxlim)
        self.ymaxvar = tk.StringVar()
        self.ymaxvar.trace("w",self.updateaxlim)
        
        self.zminvar = tk.StringVar()
        self.zminvar.trace("w",self.updatezlim)
        self.zmaxvar = tk.StringVar()
        self.zmaxvar.trace("w",self.updatezlim)
        
        l=tk.Label(self,text='xaxis:')
        l.grid(row=row,column=0)
        self.xminfield=tk.Entry(self,width=8,textvariable=self.xminvar)
        self.xminfield.grid(row=row,column=1)
        self.xmaxfield=tk.Entry(self,width=8,textvariable=self.xmaxvar)
        self.xmaxfield.grid(row=row,column=2)
        
        row+=1
        l=tk.Label(self,text='yaxis:')
        l.grid(row=row,column=0)
        self.yminfield=tk.Entry(self,width=8,textvariable=self.yminvar)
        self.yminfield.grid(row=row,column=1)
        self.ymaxfield=tk.Entry(self,width=8,textvariable=self.ymaxvar)
        self.ymaxfield.grid(row=row,column=2)
        
        row+=1
        l=tk.Label(self,text='zaxis:')
        l.grid(row=row,column=0)
        self.zminfield=tk.Entry(self,width=8,textvariable=self.zminvar)
        self.zminfield.grid(row=row,column=1)
        self.zmaxfield=tk.Entry(self,width=8,textvariable=self.zmaxvar)
        self.zmaxfield.grid(row=row,column=2)

        row+=1
        self.meshVar = tk.IntVar(value=0)
        meshbox=tk.Checkbutton(self,text='show mesh',variable=self.meshVar,command=self.mesh_callback)
        meshbox.grid(sticky = tk.W,row=row,column=0)

        self.quivVar = tk.IntVar(value=0)
        quivbox=tk.Checkbutton(self,text='show arrows',variable=self.quivVar)
        quivbox.grid(sticky = tk.W,row=row,column=1)

        self.normVar = tk.IntVar(value=0)
        normbox=tk.Checkbutton(self,text='norm arrows',variable=self.normVar)
        normbox.grid(sticky = tk.W,row=row,column=2)

        row+=1
        self.stream = tk.IntVar(value=0)
        imstream=tk.Checkbutton(self,text='stream to images :',variable=self.stream)
        imstream.grid(sticky = tk.W,row=row,column=0)
        self.picture_dir_set=False

        self.maskdry = tk.IntVar(value=1)
        maskbox=tk.Checkbutton(self,text='mask dry',variable=self.maskdry)
        maskbox.grid(sticky = tk.W,row=row,column=1)

        self.uselonlat = tk.IntVar(value=0)
        llbox=tk.Checkbutton(self,text='use lonlat',variable=self.uselonlat,command=self.lonlat_callback)
        llbox.grid(sticky = tk.W,row=row,column=2)
		
        row+=1
        h2=tk.Label(self,text='\n Extract:       ',anchor='w',font='Helvetica 10 bold')
        h2.grid(row=row,column=0)
        
        row+=1
        l7=tk.Label(self,text='extract from:')
        l7.grid(row=row,column=0)
        self.exfrom=tk.Entry(self,width=19)
        self.exfrom.grid(row=row+1,column=0,columnspan=2)
        #self.exfrom.insert(8,'0')
        self.exfrom.insert(20,str(self.dates[0])[:19])
        l8=tk.Label(self,text='extract until:')
        l8.grid(row=row,column=1)
        self.exto=tk.Entry(self,width=19)
        self.exto.grid(row=row+1,column=2,columnspan=2)
        #self.exto.insert(8,str(self.nt))
        self.exto.insert(20,str(self.dates[self.nt-1])[:19])

		#debug button
        row+=2   
        debug = tk.Button(self,text='debug',command=self.debug_callback)
        debug.grid(row=row,column=2)

		#debug button
        debug = tk.Button(self,text='close figs',wraplength=36,command=self.close_callback)
        debug.grid(row=row,column=3)


		
        # initial plot
        self.ivs=1
        self.total_time_index=0
        self.varname='depth'
        self.depths=self.ncs[self.vardict['depth']]['depth'][0,:].values
        self.nodevalues=self.ncs[self.vardict[self.varname]][self.varname][0,:].values
        self.shape=self.nodevalues.shape
        self.plot=plt.figure(self.fig0)

        print("plotting " + str(self.varname))                      
        self.clim=(np.min(self.nodevalues),np.max(self.nodevalues))       
        self.fix_clim=False
        #self.quiver=0 put at self.level tk which calls update
        self.mesh_plot=0
        self.narrows=40 # arrows shown along one axis 
        self.anno=[]
        #from IPython import embed; embed()	
        if self.use_cmocean:		
             #self.cmap0=cmo.haline
             plt.set_cmap('cmo.deep')			 
        else:
             #self.cmap0=plt.cm.jet
             plt.set_cmap(plt.cm.jet)
        self.plotx,self.ploty=self.x,self.y
        # add plot of bbathymetry		
        self.ph0,self.ch0=self.schism_plotAtelems(self.depths,add_cb=False)
        self.ph0.set_cmap('gray')  #testing
        self.ph0.set_clim((-4,4))		
        self.ph,self.ch=self.schism_plotAtelems(self.nodevalues)
        #self.cmap_callback()			 
        self.title=self.varname		
        plt.title(self.varname)
        #plt.tight_layout()
        # colormap
        if self.use_cmocean:
            self.cmap.set('deep') # default value						
        else:
            self.cmap.set('jet') # default value			
		
		
        self.plot.tight_layout()
        self.minfield.insert(8,str(self.nodevalues.min()))
        self.maxfield.insert(8,str(self.nodevalues.max()))
		# only one figure yet
        plt.figure(1).canvas.draw()
        plt.figure(1).canvas.flush_events() # flush gui events
        #self.update_plots()        # on unix cluster initial fiugre remains black -> therefore reload
        print("done initializing")                      

        #row+=1    
        # insert axlimits
        self.xminfield.insert(8,str(plt.gca().get_xlim()[0]))
        self.xmaxfield.insert(8,str(plt.gca().get_xlim()[1]))
        self.yminfield.insert(8,str(plt.gca().get_ylim()[0]))
        self.ymaxfield.insert(8,str(plt.gca().get_ylim()[1]))

    ####### call back
    def variable_callback(self,*args):    # select variable
        print("selected variable " + self.variable.get())
        self.varname=self.variable.get()
        self.shape=self.ncs[self.vardict[self.varname]][self.varname].shape        
        self.ivs=1+(self.shape[0]==2)
        self.schism_updateAtelems()

        #update extract plots in figure 2
        #if self.fig1 in plt.get_fignums():
        #        self.extract(coords=self.coords)   

		#updarte only for current period
        #self.plot_windows[fignr]['extract'](self.plot_windows[fignr]['coords'])
		# update only las figure coordinates
		# all transects and last open time series plot
        figrecord=list(self.plot_windows.keys())
        coords_re_extract=[]		
        count=0
        for fignr in figrecord[::-1]: 
            if fignr in plt.get_fignums():		
                extract=self.plot_windows[fignr]['extract']
                self.coords=self.plot_windows[fignr]['coords'].copy()
                if fignr in plt.get_fignums() and (extract==self.profiles or extract==self.transect_callback):
                    self.activefig=self.plot_windows[fignr]['nr']			
                    self.xs=self.plot_windows[fignr]['xs'].copy()
                    extract(self.plot_windows[fignr]['coords'])
                    count+=1
                    #print(count)					
				# for time series redp only ones	
                elif len(coords_re_extract)==0 or (self.coords not in coords_re_extract):
                    extract(self.plot_windows[fignr]['coords'])
                    coords_re_extract.append(self.coords)
                    count+=1
                    #print(count)
                if count==1:
                    print('extracting variable at coordinates corresponding to open figures')				
                #break
			
		# multifig
       #figrecord=list(self.plot_windows.keys())
       ##from IPython import embed; embed()
       ##coords=[self.plot_windows[fignr]['coords'] for fignr in figrecord]
       ##setups=[self.plot_windows[fignr]['setup'] for fignr in figrecord if fignr in plt.get_fignums()]
       ##varnames=[self.plot_windows[fignr]['varname'] for fignr in figrecord if fignr in plt.get_fignums()]
		## if variable not already plotted for coordinates setup and diff type			
       ##self.plot_windows[self.fignums]['diff']		
		#
		## check to prevent duuplicate figures
		## check if in line loops faster or tracked dictionary
       #setups=[]
       #varnames=[] 
       #diffs=[]
       ##diffi=self.CheckDiff.get()
       #coords=[]
       #for fignr in figrecord: 
       #    if fignr in plt.get_fignums():
       #        setups.append(self.plot_windows[fignr]['setup'] )
       #        varnames.append(self.plot_windows[fignr]['varname'])
       #        diffs.append(self.plot_windows[fignr]['diff'])
       #        coords.append(self.plot_windows[fignr]['coords'])
       #setups=np.asarray(setups)
       #varnames=np.asarray(varnames)
       #diffs=np.asarray(diffs)
       #coords=np.asarray(coords)
		#		
		#		
       ## update multiple figures or only one? I choose only one for current variable now. all now				
       #for fignr in figrecord: #self.plot_windows.keys():
       #    #if fignr in plt.get_fignums():
       ## if variable not already plotted for coordinates setup and diff type							
       ##if (True not in (np.asarray(varnames)==self.varname)	& (np.asarray(setups)==self.active_setup) & (np.asarray(diffs)==diffi)):		
       #    self.activefig=self.plot_windows[fignr]['nr']
       #    self.xs=self.plot_windows[fignr]['xs'].copy()
       #    self.coords=self.plot_windows[fignr]['coords'].copy()
       #    diffi=self.plot_windows[fignr]['diff']
       #    if (fignr in plt.get_fignums()) and ( True not in ( (varnames==self.varname)	& (setups==self.active_setup) & (coords==self.coords) & (diffs==diffi))):	
       #        self.plot_windows[fignr]['extract'](self.plot_windows[fignr]['coords'])
				
    def variable_selection(self,event): #combobox callback done differntly
        self.variable.set(self.combo.get())        #selection = self.combo.get()
				
    def lvl_callback(self,*args):     # set level    
        self.lvl=self.lvl_tk.get()
        print("selected level " +  str(self.lvl))
        self.schism_updateAtelems()
		
    def stp_callback(self,*args): # select setip # 
        self.active_setup=self.stp_tk.get()

        if self.CheckDiff.get()==0:
            print("selected setup " +  str(self.active_setup))
            self.ncs=self.nclist[self.active_setup]
        else:	
            print("selected setup difference " +  str(self.active_setup) + ' - 0') 
            self.ncs=self.diffnclist[self.active_setup-1]
        self.nt=self.nts[self.active_setup]
        self.variable_callback()
	
    def stack_callback(self,*args): # load new stack # 
        stacknow=self.stack.get()
        if stacknow!=self.current_stack:
            if  self.stacks[self.nstacks-1] < stacknow or stacknow< self.stacks[0]:
                print('stack not existent')
            else:
                self.file=self.combinedDir+'schout_'+str(stacknow)+'.nc' #self.files[self.stack.get()-1] 
                print("loading" +  self.file)
                self.ti_tk.set(0) 
                self.ti=0
                self.current_stack=stacknow

    def ti_callback(self,*args): #set timestep
        self.ti=self.ti_tk.get()
        print("selected timestep " +  str(self.ti))
        self.total_time_index=self.ti+self.stack_size*(self.current_stack-self.stack0) 
		
        self.dryelems=self.ncs[self.filetag][self.dryvarname][self.total_time_index,:][self.origins]
        if (self.drynodename == False) and (self.oldio==True):
            self.drynodes=self.ncs[self.vardict['hvel']]['hvel'][0,self.total_time_index,:,-1].values==0
        else:
            self.drynodes=np.asarray(self.ncs[self.filetag][self.drynodename][self.total_time_index,:],bool)	
        self.schism_updateAtelems()

        for fignr in self.plot_windows.keys():
            extract=self.plot_windows[fignr]['extract']
            if fignr in plt.get_fignums() and (extract==self.profiles or extract==self.transect_callback):
                self.activefig=self.plot_windows[fignr]['nr']
                self.xs=self.plot_windows[fignr]['xs'].copy()
                self.coords=self.plot_windows[fignr]['coords'].copy()
                extract(self.plot_windows[fignr]['coords'])
            else:        
                self.plot_windows.pop(fignr)

    def playcallback(self,*args):
            count=(self.current_stack-self.stacks[0])*self.stack_size+self.ti_tk.get()
            if count < self.stack_size*len(self.files)-1 and self.play:
                self.nextstep()
                if self.stream.get():
                    self.stream2image()
            else:
                self.play=True
                return
            self.master.after(1,self.playcallback) #sleep was 5 now check 1 create recursive play loop
         	
    def stopcallback(self,*args):
         self.play=False          
                            
    def debug_callback(self,*args):
         print('open command variables accessible self. exit interactive mode via exit())') 
         from IPython import embed; embed()	

    def close_callback(self,*args):
        for fignr in list(plt.get_fignums())[1:]:		 
            plt.close(fignr)
            self.pt0=0
    def get_layer_weights(self,dep): 
        print('calculating weights for vertical interpolation')
        ibelow=np.zeros(self.nnodes,int)
        iabove=np.zeros(self.nnodes,int)
        weights=np.zeros((2,self.nnodes))
		
        # garbage values below ibtm different type , nan or strange values wrong values
        zcor=self.ncs[self.vardict[self.zcorname]][self.zcorname][self.total_time_index,:,:].values#self.ti_tk.get() #self.ncv['zcor']
        zcor=np.ma.masked_array(zcor,mask=self.mask3d)
		
        a=np.sum(zcor<=dep,1)-1
        #ibelow=a+np.sum(zcor.mask,1)-1
        ibelow=a+self.ncs[self.filetag][self.bindexname][0,:].values-1
        #ibelow=a+(self.nc['bottom_index_node'][:]-1)-1
        iabove=np.minimum(ibelow+1,self.nz-1)
        inodes=np.where(a>0)[0]
        ibelow2=ibelow[inodes]
        iabove2=iabove[inodes]

        d2=zcor[inodes,iabove2]-dep
        d1=dep-zcor[inodes,ibelow2]
        ivalid=d1>0.0
        iset=d1==0.0
        d1=d1[ivalid]
        d2=d2[ivalid]
        weights[0,inodes[ivalid]]=1-d1/(d1+d2)#1/d1/(1/d1+1/d2)
        weights[1,inodes[ivalid]]=1-weights[0,inodes[ivalid]]#1/d2/(1/d1+1/d2)
        weights[0,inodes[iset]]=1
        weights[:,np.sum(weights,0)==0.0]=np.nan
        
        return ibelow, iabove, weights

                
    def stream2image(self,*args):
        if self.picture_dir_set==False:
            self.streamdir=filedialog.askdirectory(title='select output directory for image stream')+'/'
            self.picture_dir_set=True
        for nr,fig in enumerate(plt.get_fignums()):
            plt.figure(fig)
            plt.savefig(self.streamdir+'_{:d}_'.format(nr)+'{0:05d}'.format(self.total_time_index)+'_'+self.varname+'.png',dpi=300)
          
    def cmap_callback(self,*args):   
        if self.use_cmocean:
               #cmap='cmo.{:s}'.format(self.cmap.get())# # works
               cmap=eval('cmo.{:s}'.format(self.cmap.get()))
               cmap.set_bad('None') # None for transparent
               #cmap.set_bad('gray') # None for transparent
               cmap.set_over('purple')
               cmap.set_under('black')
               cmap.name='cmo.{:s}'.format(self.cmap.get())
               self.ph.set_cmap(cmap)			                  

        else: #matplotlib
               cmap=plt.cm.get_cmap(self.cmap.get())
               cmap.set_bad('None')			   
               #cmap.set_bad('gray')
               cmap.set_over('purple')
               cmap.set_under('black')
               self.ph.set_cmap(cmap)				   
        for i in plt.get_fignums():#reversed(plt.get_fignums()[:-1]):
               plt.figure(i) 
               plt.set_cmap(cmap)
        self.update_plots() #self.plot.canvas.draw() #self.update_plots()
        return cmap	         		
		
    def mesh_callback(self,*args):        
        """ add mesh to plot """
        plt.figure(self.fig0)
        if (self.meshVar.get()==1) and (self.mesh_plot==0):
            self.mesh_plot=1
            plt.gca().add_collection(self.tripc)
            if self.hasquads:
                plt.gca().add_collection(self.quadpc)
        elif  (self.meshVar.get()==0) and (self.mesh_plot!=0):
            self.tripc.remove()
            if self.hasquads:
                self.quadpc.remove()
            self.mesh_plot=0
        self.update_plots()  

    # this was embedded in stp clalback and variable trace
    def CheckDiff_callback(self,*args):        
        """ toggle plot setup or differences """
        if (self.CheckDiff.get()==1): 
            print('plotting differences setup {:d} - setup 0'.format(self.active_setup) )		
            self.ncs=self.diffnclist[self.active_setup-1]
            #self.cmap.set() depends on cmocean or not
        elif  (self.CheckDiff.get()==0): 
            self.ncs=self.nclist[self.active_setup]
            print('plotting  setup {:d} '.format(self.active_setup) )		
            #self.cmap.set()			
 
    def lonlat_callback(self,*args):     
        if self.uselonlat.get()==1:
            self.plotx,self.ploty=self.lon,self.lat
        else:
            self.plotx,self.ploty=self.x,self.y		
        plt.figure(self.fig0)
        plt.clf()
        self.ph0,self.ch0=self.schism_plotAtelems(self.depths,add_cb=False)
        self.ph0.set_array(self.depths[self.faces[:,:3]].mean(axis=1))  # noetig ? workaround	
        self.ph0.set_cmap('gray')  #testing ad bathymetry
        self.ph0.set_clim((-4,4))		
        self.ph,self.ch=self.schism_plotAtelems(self.nodevalues)
        self.schism_updateAtelems()
		
    #def Tavg_callback(self,*args): 
    #    if self.uselonlat.get()==1:
    #        self.plotx,self.ploty=self.lon,self.lat
    #    else:
    #        self.plotx,self.ploty=self.x,self.y		
	
		
    def ask_coordinates(self,n=-1,coords=None,interp=False):
            plt.figure(self.fig0)
            if coords==None:	 		
                plt.title("click coordinates in Fig.1. Press ESC when finished")
                print("click coordinates in Fig.1. Press ESC when finished")
                self.pt0i=self.pt0 # keep track of point annotations			
                self.update_plots()
                plt.figure(self.fig0)
                self.coords=plt.ginput(n,show_clicks='True')
                plt.title(self.titlegen(self.lvl))
            else:
            	self.coords=coords
            	self.pt0i=self.plot_windows[self.activefig]['p0']
				
            if interp:  # interpolate coordinates for transect
			           
            	xy=np.asarray(self.coords)
            	dxdy=np.diff(xy,axis=0)
            	dr=np.sqrt((dxdy**2).sum(axis=1))
            	r=dxdy[:,1]/dxdy[:,0]
            	
            	
            	#self.minavgdist=np.float(input('Enter distance [m ord degree if ics=2] for between point interpolation:'))
            	self.minavgdist=tk.simpledialog.askfloat(title='interp dx', prompt='Enter distance [hgrid.gr3 units i.e. m] for between point interpolation: [0:= keep points]',initialvalue=250,minvalue=0)
            	coords1=[] 
            	if self.minavgdist !=0:
            		for i in range(len(self.coords)-1):
            			dx=np.linspace(0,dxdy[i,0],int(np.floor(dr[i]/self.minavgdist)))
            			coords1+=([(xi[0][0],xi[0][1]) for xi in zip((xy[i,:]+np.stack((dx, dx*r[i]),axis=1)))])
            		self.coords=coords1											
            	else:		
            		coords1=self.coords					
            else:
            	coords1=self.coords
				
            if self.uselonlat.get()==0:			
            	d,self.nn=self.xy_nn_tree.query(self.coords)  # nearest node
            else:								
            	d,self.nn=self.ll_nn_tree.query(self.coords)  # nearest node				
				

            ivalid=~np.isnan(np.asarray(coords1).sum(axis=1))	
            self.npt=len(self.nn)							
            print('{:d} nn points are NaN and removed from list and plotting'.format(len(ivalid)-ivalid.sum()))				                
            xy=np.asarray(coords1)[ivalid,:]  # remove nan coords
            coords1=list(zip(xy[:,0],xy[:,1]))
            x,y=xy[:,0],xy[:,1]
            self.npt=ivalid.sum()#len(x)
            self.xs=np.tile(np.arange(self.npt),(self.nz,1)).T
            self.coords=coords1					
			
            if (self.extract==self.transect_callback) or (self.extract==self.hovmoeller_horz): # interp coordinates
			
                #xy=np.asarray(coords1)
                #x,y=xy[:,0],xy[:,1]
                #self.npt=len(x)
                #self.coords=coords1
                #print('interpolate node weights from parents')				
				# usin nn instead
                #self.parents,self.ndeweights=self.find_parent_tri(self.faces,self.x,self.y,xy[:,0],xy[:,1],dThresh=self.maxavgdist*3)
                #print('done interpolating node weights from parents')				
                #self.xs=np.tile(np.arange(len(self.parents)),(self.nz,1)).T
				
                #self.anno=plt.plot(x,y,'k.-')        
                #self.anno.append(plt.text(x[0],y[0],'P '+str(0)))
                #for ix in range(25,len(x),25):
                #    self.anno.append(plt.text(x[ix],y[ix],'P '+str(ix)))
                #self.anno.append(plt.text(x[0],y[0],'P '+str(0)))
                #self.anno.append(plt.text(x[-1],y[-1],'P '+str(self.npt-1)))

                anno=plt.plot(x,y,'k.-')        
                anno.append(plt.text(x[0],y[0],'P '+str(self.pt0i+0)))
                for ix in range(25,len(x),25):
                    anno.append(plt.text(x[ix],y[ix],'P '+str(self.pt0i+ix)))
                anno.append(plt.text(x[0],y[0],'P '+str(0)))
                anno.append(plt.text(x[-1],y[-1],'P '+str(self.pt0i+self.npt-1)))

            else:    
                if self.uselonlat.get()==0:			
                    d,self.nn=self.xy_nn_tree.query(self.coords)  # nearest node
                else:								
                    d,self.nn=self.ll_nn_tree.query(self.coords)  # nearest node				
                self.npt=len(self.nn)
                #self.anno=plt.plot(self.plotx[self.nn],self.ploty[self.nn],'k+')             
                anno=plt.plot(self.plotx[self.nn],self.ploty[self.nn],'k+')             				
                for i,coord in enumerate(self.coords):
                    xi,yi=coord
                    #self.anno.append(plt.text(xi,yi,'P '+str(i)))
                    anno.append(plt.text(xi,yi,'P '+str(self.pt0i+i)))					
                if self.npt==1:
                    self.nn=self.nn[0]

            if coords==None:					
                self.pt0+=self.npt		
            print('done interpolating transect coordinates')
			
	
            self.update_plots()

			# initilaize new plot window	
            self.fignums+=1
            plt.get_fignums()[1:]
            fig2=plt.figure(self.fignums)
            nr=self.fignums
            self.plot_windows[self.fignums]={'fh':fig2,'nr':nr,'anno':anno,'type':self.timeseries,'coords':self.coords.copy(),'extract':self.extract,'xs':self.xs.copy()}
            self.plot_windows[self.fignums]['p0']=self.pt0i
            self.plot_windows[self.fignums]['varname']=self.varname
            self.plot_windows[self.fignums]['setup']=self.active_setup
            self.plot_windows[self.fignums]['diff']=self.CheckDiff.get()
            self.activefig=nr
			
    def read_time_selection(self):
        np.datetime64(self.exfrom.get())-self.dates[0]
        i0=np.int((np.datetime64(self.exfrom.get())-self.dates[0])/self.dt)
        i1=np.int((np.datetime64(self.exto.get())-self.dates[0])/self.dt)
        return i0,i1
			
    def timeseries(self,coords=None):
        """"
        Extract timeseries at nextneighbours to clicked coordinates
        """
        
        self.extract=self.timeseries # identify method
        if coords==None:
            self.ask_coordinates(n=-1,coords=None)
        else:	
            self.ask_coordinates(n=-1,coords=self.coords,interp=False)	

        print('extracting timeseries for ' + self.varname + ' at coordinates: ' + str(self.coords))
        #i0,i1=int(self.exfrom.get()),int(self.exto.get())
        #from IPython import embed; embed()
        #i0,i1=int(self.exfrom.get()),int(self.exto.get())
        i0,i1=self.read_time_selection()
        if self.oldio:
            self.t=self.ncs['schout']['time'].values[i0:i1]
        elif self.newio<2:
            self.t=np.asarray([self.reftime + dt.timedelta(seconds=ti) for ti in            self.ncs[self.filetag]['time'].values],np.datetime64)[i0:i1]
        else:	
            self.t=self.ncs[self.filetag]['time'].values[i0:i1]
        
        if self.shape[1:]==(self.nnodes,self.nz): #self.shape==(self.nt,self.nnodes,self.nz): #3D
            self.ncs[self.vardict[self.varname]][self.varname][:,self.nn,self.lvl]
            self.ts=self.ncs[self.vardict[self.varname]][self.varname][i0:i1,self.nn,self.lvl]
        elif self.shape[1:]==(self.nnodes,): #self.shape==(self.nt,self.nnodes):
            self.ts=self.ncs[self.vardict[self.varname]][self.varname][i0:i1,self.nn]
        #elif self.shape==(self.nt,self.nnodes,2)
        elif self.shape[2:]==(self.nnodes,):#self.shape==(2,self.nt,self.nnodes):
            if self.npt==1:
                self.ts=self.ncs[self.vardict[self.varname]][self.varname][:,i0:i1,self.nn].values.reshape(2,i1-i0,1)
            else:
                self.ts=self.ncs[self.vardict[self.varname]][self.varname][:,i0:i1,self.nn].values
        elif self.shape[2:]==(self.nnodes,self.nz):#self.shape==(2,self.nt,self.nnodes,self.nz):
            if self.npt==1:
                self.ts=self.ncs[self.vardict[self.varname]][self.varname][:,i0:i1,self.nn,self.lvl].values.reshape(2,i1-i0,1)
            else:
                self.ts=self.ncs[self.vardict[self.varname]][self.varname][:,i0:i1,self.nn,self.lvl].values

        prefix=['',' $\Delta$ '][ self.CheckDiff.get()]		
        if self.ivs==1:
            if self.CheckEval.get()!=0:
                expr= self.evalex.get()
                expr=expr[expr.index('=')+1:].replace('x','self.ts').replace('A','self.A').replace('dt','self.dt')
                self.ts=eval(expr)  
            plt.plot(self.t,self.ts)#/86400
            plt.xlabel('time')
            plt.ylabel(prefix+self.varname)
            plt.grid()
            self.lh=plt.legend(['P'+str(i+self.pt0i) for i in range(self.npt)],loc='upper center',bbox_to_anchor=(0.5, 1.02),ncol=6)
        else:
            comps=[' - u', '- v ', '- abs' ]
            for iplt in range(1,3):
                plt.subplot(3,1,iplt)
                plt.plot(self.t,self.ts[iplt-1,:])#/86400
                plt.tick_params(axis='x',labelbottom='off')
                plt.ylabel(self.varname + comps[iplt-1])
                plt.grid()
                if iplt==1:
                    plt.legend(['P'+str(i+self.pt0i) for i in range(self.npt)],loc='upper center',bbox_to_anchor=(0.5, 1.3),ncol=6)

            if self.CheckEval.get()!=0:
                    expr= self.evalex.get()
                    expr=expr[expr.index('=')+1:].replace('x','ts').replace('A','self.A').replace('dt','self.dt')
                    self.nodevalues=eval(expr)    

            plt.subplot(3,1,3)
            plt.plot(self.t,np.sqrt(self.ts[0,:]**2+self.ts[1,:]**2))#/86400
            plt.xlabel('time')
            plt.ylabel(prefix+self.varname + comps[2])
            plt.grid()
        
        plt.gcf().autofmt_xdate() # format date
        plt.tight_layout()
        self.update_plots()
		
        if len(self.nclist) > 1:
            plt.title('config ' + str(self.active_setup))		
        print("done extracting time series")                                     

    def profiles(self,coords=None):
        """" Extract profiles at nextneighbours to clicked coordinates """
        self.extract=self.profiles
        if coords==None:
            self.ask_coordinates()
			
        zs=self.ncs[self.zcorname][self.zcorname][self.total_time_index,self.nn,:].values                        
        if self.shape[1:]==(self.nnodes,self.nz):#self.shape==(self.nt,self.nnodes,self.nz):
            ps=self.ncs[self.vardict[self.varname]][self.varname][self.total_time_index,self.nn,:].values

            if self.npt>1:
                ps=ps.swapaxes(0,1)
                zs=zs.swapaxes(0,1)			
            
        elif self.shape[2:]==(self.nnodes,self.nz):#self.shape==(2,self.nt,self.nnodes,self.nz): #hvel
            ps=self.ncs[self.vardict[self.varname]][self.varname][:,self.total_time_index,self.nn,:].values
            if self.npt>1:
                ps=ps.swapaxes(1,2)
                zs=zs.swapaxes(0,1)			

        else:
            print("variable has no depth associated")
            return
            
        fig2=plt.figure(self.activefig)#plt.figure(self.fig1)
        fig2.clf()
        if self.ivs==1:
            plt.title(self.titlegen(''))
            #plt.title(str(self.reftime + dt.timedelta(seconds=int(self.ncs[self.filetag]['time'][self.total_time_index].values))))
            plt.plot(ps,zs)
            plt.ylabel('depth / m')
            plt.xlabel(self.varname)
            plt.legend(['P'+str(i+self.pt0) for i in range(self.npt)])
            plt.grid()
        else:
            comps=[' - u', '- v ', '- abs' ]
            for iplt in range(1,3):
                plt.subplot(1,3,iplt)
                plt.plot(ps[iplt-1,:],zs)
                plt.grid()
                plt.xlabel(self.varname + comps[iplt-1])
                if iplt==1:
                    if self.newio<2:				
                        plt.title(str(self.reftime + dt.timedelta(seconds=int(self.ncs[self.filetag]['time'][self.total_time_index].values))))
                    else:	
                        plt.title(self.ncs[self.filetag]['time'][self.total_time_index].values)					
                    plt.legend(['P'+str(i+self.pt0) for i in range(self.npt)])
                    plt.ylabel('depth / m')
                else:
                    plt.tick_params(axis='y',labelleft='off')
            plt.subplot(1,3,3)
            plt.plot(np.sqrt(ps[0,:]**2+ps[1,:]**2),zs)
            plt.xlabel(self.varname + comps[2])
            plt.grid()
            plt.tick_params(axis='y',labelleft='off')
            plt.tight_layout()
            
        self.zminfield.delete(0, 'end')
        self.zmaxfield.delete(0, 'end')
        self.zminfield.insert(8,str(plt.gca().get_ylim()[0]))
        self.zmaxfield.insert(8,str(plt.gca().get_ylim()[1]))
        if len(self.nclist) > 1:
            plt.title('config ' + str(self.active_setup))		
        self.update_plots()

    def hovmoeller(self,coords=None):
        
        self.extract=self.hovmoeller
        if coords==None:
            self.ask_coordinates(n=1)

        print('extracting hovmoeller for ' + self.varname + ' at coordinates: ' + str(self.coords))
        #i0,i1=int(self.exfrom.get()),int(self.exto.get())
        i0,i1=self.read_time_selection()
        self.zcor=np.squeeze(self.ncs[self.vardict[self.zcorname]][self.zcorname][i0:i1,self.nn,:])

        if self.oldio:
            self.t=self.ncs['schout']['time'].values[i0:i1]
        elif self.newio<2:
            self.t=np.asarray([self.reftime + dt.timedelta(seconds=ti) for ti in            self.ncs[self.filetag]['time'].values],np.datetime64)[i0:i1]
        else:	
            self.t=self.ncs[self.filetag]['time'].values[i0:i1]

		
        if self.shape[1:3]==(self.nnodes,self.nz):#self.shape[:3]==(self.nt,self.nnodes,self.nz):
            self.ts=np.squeeze(self.ncs[self.vardict[self.varname]][self.varname][i0:i1,self.nn,:])
        elif self.shape[2:]==(self.nnodes,self.nz):#self.shape[1:]==(self.nt,self.nnodes,self.nz):#hvel
            self.ts=np.squeeze(self.ncs[self.vardict[self.varname]][self.varname][:,i0:i1,self.nn,:])
        elif self.shape[1:]==(self.nnodes,) or self.shape[1:]==(self.nnodes,2):#self.shape==(self.nt,self.nnodes) or self.shape==(self.nt,self.nnodes,2):
            print("variablae has no depths")
            return

        #fig2=plt.figure(self.fig1)
        fig2=plt.figure(self.activefig)
        fig2.clf()
        if self.ivs==1:
            plt.pcolor(np.tile(self.t,(self.nz,1)).transpose(),self.zcor,self.ts)#/86400
            plt.ylabel('depth')
            plt.title(self.varname)
        else:
            comps=[' - u', '- v ', '- abs' ]
            for iplt in range(1,3):
                plt.subplot(3,1,iplt)
                plt.pcolor(np.tile(self.t,(self.nz,1)).transpose(),self.zcor,np.squeeze(self.ts[iplt-1,:,:]))#/86400#squeeze(self.ts[:,:,iplt-1]))#/86400
                plt.ylabel('depth')
                plt.title(self.varname)
                plt.colorbar()
                plt.title(self.varname + comps[iplt-1])
                plt.tick_params(axis='x',labelbottom='off')
                plt.set_cmap(self.cmap.get())
            plt.subplot(3,1,3)
            plt.pcolor(np.tile(self.t,(self.nz,1)).transpose(),self.zcor,np.sum(np.sqrt(self.ts**2),axis=0))#/86400
            plt.title(self.varname + comps[2])  

        plt.colorbar()
        plt.xlabel('time')
        plt.gcf().autofmt_xdate() # format date		
        plt.tight_layout()

        self.zminfield.delete(0, 'end')
        self.zmaxfield.delete(0, 'end')
        self.zminfield.insert(8,str(plt.gca().get_ylim()[0]))
        self.zmaxfield.insert(8,str(plt.gca().get_ylim()[1]))
        self.cmap_callback()# plot with chosen colormap
        if len(self.nclist) > 1:
            plt.title('config ' + str(self.active_setup))

    def hovmoeller_horz(self,coords=None):
        
        self.extract=self.hovmoeller_horz
        if coords==None:
            self.ask_coordinates(n=-1,interp=True)

        print('extracting hovmoeller for ' + self.varname + ' at coordinates: ' + str(self.coords))
        #i0,i1=int(self.exfrom.get()),int(self.exto.get())
        i0,i1=self.read_time_selection()
		#self.zcor=np.squeeze(self.ncs[self.vardict[self.zcorname]][self.zcorname][i0:i1,self.nn,:])
        if self.oldio:
            self.t=self.ncs['schout']['time'].values[i0:i1]
        elif self.newio<2:
            self.t=np.asarray([self.reftime + dt.timedelta(seconds=ti) for ti in            self.ncs[self.filetag]['time'].values],np.datetime64)[i0:i1]
        else:	
            self.t=self.ncs[self.filetag]['time'].values[i0:i1]			
			
        if self.shape[1:3]==(self.nnodes,self.nz):#self.shape[:3]==(self.nt,self.nnodes,self.nz):
            self.ts=np.squeeze(self.ncs[self.vardict[self.varname]][self.varname][i0:i1,self.nn,self.lvl])
        elif self.shape[2:]==(self.nnodes,self.nz):#self.shape[1:]==(self.nt,self.nnodes,self.nz):#hvel
            self.ts=np.squeeze(self.ncs[self.vardict[self.varname]][self.varname][:,i0:i1,self.nn,self.lvl])
        elif self.shape[1:]==(self.nnodes):#self.shape==(self.nt,self.nnodes):
            self.ts=np.squeeze(self.ncs[self.vardict[self.varname]][self.varname][i0:i1,self.nn])
        elif self.shape[1:]==(self.nnodes,2):#self.shape==(self.nt,self.nnodes,2):
            self.ts=np.squeeze(self.ncs[self.vardict[self.varname]][self.varname][i0:i1,self.nn,:])		
        else:
            print('variable has non suitable format')
		
        self.xs=np.tile(np.arange(self.npt),(len(self.t),1))#.T
		
        fig2=plt.figure(self.activefig)
        fig2.clf()
        if self.ivs==1:
            plt.pcolormesh(np.tile(self.t,(len(self.nn),1)).transpose(),self.xs,self.ts)#/86400
            plt.ylabel('point')
            plt.xlabel('time')
            plt.title(self.varname)
        else:
            comps=[' - u', '- v ', '- abs' ]
            for iplt in range(1,3):
                plt.subplot(3,1,iplt)
                plt.pcolormesh(np.tile(self.t,(self.nz,1)).transpose(),self.xs,np.squeeze(self.ts[iplt-1,:,:]))#/86400#squeeze(self.ts[:,:,iplt-1]))#/86400
                plt.ylabel('depth')
                plt.title(self.varname)
                plt.colorbar()
                plt.title(self.varname + comps[iplt-1])
                plt.tick_params(axis='x',labelbottom='off')
                plt.set_cmap(self.cmap.get())
            plt.subplot(3,1,3)
            plt.pcolor(np.tile(self.t,(self.nz,1)).transpose(),self.zcor,np.sum(np.sqrt(self.ts**2),axis=0))#/86400
            plt.title(self.varname + comps[2])  

        plt.colorbar()
        plt.xlabel('time')
        plt.gcf().autofmt_xdate() # format date		
        if len(self.nclist) > 1:
            plt.title('config ' + str(self.active_setup))
        plt.tight_layout()
        self.cmap_callback()# plot with chosen colormap

    def plot_transect(self,dataTrans,is2d=False):		
        # dry check nn interp
        plt.figure(self.activefig)
        #plt.clf()
        d,qloc=self.xy_nn_tree.query(self.coords)
        isdry=np.isin(qloc,self.faces[self.dryelems==1,:3])
        if is2d:
            plt.figure(self.activefig)
            dataTrans=np.ma.masked_array(dataTrans,mask=dataTrans.mask | isdry)
            plt.plot(self.xs[:,-1],dataTrans) 
            plt.ylabel(self.varname)
            plt.grid()            
            ch=plt.gca()
        else:		
			#self.zi[:,-1]==self.zi[:,-2]
            dataTrans=np.ma.masked_array(dataTrans,mask=dataTrans.mask | np.tile(isdry,(self.zi.shape[1],1)).T)
            self.zi.mask[np.isnan(self.zi)]=True
            cmap=self.cmap_callback() #        self.set_colormap_workaround(self)			
            plt.figure(self.activefig)
            ph=plt.pcolor(self.xs,self.zi,dataTrans,cmap=cmap) 
            plt.ylabel('depth / m')
            plt.plot(self.xs[:,0],self.zi[:,0],'k',linewidth=2)
            ch=plt.colorbar()
            ch.set_label(self.varname)
            if self.meshVar.get():
                plt.plot(self.xs,self.zi,'k',linewidth=0.2)
        if len(self.nclist) > 1:
            plt.title('config ' + str(self.active_setup)+self.titlegen(np.nan))				
        else:	
            plt.title(self.titlegen(np.nan))     
       		
        return ph,ch
		
	#def add_quiv(self,dataTrans):	
        #plt.quiver(self.xs,self.zi,dataTrans[:,:,0],dataTrans[:,:,0],color='w') #shading=['flat','faceted'][self.meshVar.get()]
    def bptransect_callback(self,coords=None):
        filename=filedialog.askopenfilename(title='opeb build point file',initialdir=self.runDir,filetypes = (('bp files', '*.bp'),('All files', '*.*')))
        m=np.loadtxt(filename,skiprows=2)[:,1:]
        self.coords=list(zip(m[:,0],m[:,1]))
        self.extract=self.transect_callback
        self.ask_coordinates(n=-1,coords=self.coords,interp=True)
        self.transect_callback(coords=self.coords)
		
    def transect_callback(self,coords=None):
        self.extract=self.transect_callback #'transect'
        if coords==None:
            self.ask_coordinates(interp=True)
            #self.zi=np.ma.masked_array(np.zeros((len(self.parents),self.nz)),mask=False)#*np.nan
        #w=self.ndeweights
        is2d = not 'nSCHISM_vgrid_layers'  in  self.ncs[self.vardict[self.varname]][self.varname].dims
        zcor=self.ncs[self.vardict[self.zcorname]][self.zcorname][self.total_time_index,:].values
        mask=self.mask3d  | np.isnan(zcor)
        zcor=np.ma.masked_array(zcor,mask=mask)
		#self.depths
        if is2d:	
            mask=mask[:,-1]		
        if self.ivs==1:
            data=self.ncs[self.vardict[self.varname]][self.varname][self.total_time_index,:].values
            data=np.ma.masked_array(data,mask=mask)#self.mask3d
        elif self.ivs==2: #ivs2=  hvel
            if is2d:	
                mask_hvel=self.mask_hvel[:,:,-1]   
            else:	
                mask_hvel=self.mask_hvel				
            data=self.ncs[self.vardict[self.varname]][self.varname][:,self.total_time_index,:].values
            data=np.ma.masked_array(data,mask=mask_hvel | np.tile(mask,(2,1,1)))
        m_interp='nn' # 'dinv'	 #interpolation method
        if m_interp=='dinv':
            self.zi=np.ma.masked_array(np.zeros((len(self.parents),self.nz)),mask=False)#*np.nan
            w=self.ndeweights
            for i in range(self.nz):
                w=np.ma.masked_array(self.ndeweights,mask=zcor[:,i][self.faces[self.parents]].mask)
                self.zi[:,i]=(zcor[:,i][self.faces[self.parents]]*w).sum(axis=1)#nan in output, not hotstart
        else:
            d,qloc=self.xy_nn_tree.query(self.coords)
            isub=np.unique(qloc,return_index=True)[1] # only use unique values # changes orde
            qloc=np.asarray([indi for indi in qloc if indi in qloc[isub]])
            self.zi=zcor[qloc,:]
		
        ylim=((np.nanmin(self.zi),5)) # nan
        plt.figure(self.activefig)
        plt.clf()
        if self.ivs==1: # scalar # interpolation along sigma layers
            
            if m_interp=='dinv':
			
                self.dataTrans=np.zeros((len(self.parents),self.nz))
                for i in range(len(self.parents)):
                    depths=zcor[self.faces[self.parents[i],:],:]
                    lvldata=data[self.faces[self.parents[i],:],:]
                    ibttm=self.ibttms[i]
                    ibttm=np.isnan(depths).sum(axis=1).max()
                    self.dataTrans[i,ibttm:]=scipy.interp(self.zi[i,ibttm:],depths[0,ibttm:],lvldata[0,ibttm:])*w[i,0] + \
                    scipy.interp(self.zi[i,ibttm:],depths[1,ibttm:],lvldata[1,ibttm:])*w[i,1] + \
                    scipy.interp(self.zi[i,ibttm:],depths[2,ibttm:],lvldata[2,ibttm:])*w[i,2]	
                    self.dataTrans[i,-1]=(lvldata[:,-1]*w[i,:]).sum() 
                    self.dataTrans[i,ibttm]=(lvldata[:,ibttm]*w[i,:]).sum() 
                self.dataTrans=np.ma.masked_array(self.dataTrans,mask=self.zi.mask)
                    #self.zi=np.ma.masked_array(self.dataTrans,mask=self.dataTrans.mask)
            else: #nearest neighbour	 interp
                
                print('usig nearest neighbour interpolation for transect')
                #d,qloc=self.xy_nn_tree.query(self.coords)
                #isub=np.unique(qloc,return_index=True)[1] # only use unique values
                #qloc=np.asarray([indi for indi in qloc if indi in qloc[isub]])
                if is2d:
                    #self.dataTrans=data[qloc[isub]]		
                    self.dataTrans=data[qloc]							
                else:
                    #self.dataTrans=data[qloc[isub],:]
                    self.dataTrans=data[qloc,:]					
                #self.xs=self.xs[:len(isub),:]					
                self.xs=self.xs[:len(qloc),:]									
                #self.dataTrans=np.ma.masked_array(self.dataTrans,mask=mask[qloc[isub]])#self.mask3d                           
                self.dataTrans=np.ma.masked_array(self.dataTrans,mask=mask[qloc])#self.mask3d                           				
                #self.zi=zcor[qloc[isub],:]
                self.zi=zcor[qloc,:]
				
				#brauch ich das?  erledigt mit mask?
                #for i in range(self.zi.shape[0]):				                
                #    self.zi[i,:self.ibbtm[qloc[isub][i]]]=self.zi[i,self.ibbtm[qloc[isub][i]]]
                #self.zi.mask=False

            if is2d:
                ylim=((np.nanmin(self.dataTrans),np.nanmax(self.dataTrans))) # nan
            else:
                ylim=((np.nanmin(self.zi),5)) # nan

            self.plot_transect(self.dataTrans,is2d)
        else: # vector
            #self.dataTrans=np.zeros((self.ivs+1,len(self.parents),self.nz)) #self.ncs[self.vardict[self.varname]][self.varname][:,i0:i1,self.nn,self.lvl]
            vert=self.ncs[self.vertvelname][self.vertvelname][self.total_time_index,:]
            vert=np.ma.masked_array(vert,mask=self.mask3d)
            data=np.concatenate((data,vert.reshape((1,self.nnodes,self.nz))),axis=0) # stack vertil
            #data=np.concatenate((data,vert.reshape((1,self.nnodes,self.nz))),axis=2) # stack vertil velocity
            #for i in range(len(self.parents)):
				# invdist  
                #depths=zcor[self.faces[self.parents[i],:],:]
                #lvldata=data[self.faces[self.parents[i],:],:]
                #ibttm=(self.nc['bottom_index_node'][self.faces[self.parents[i],:]]).max()-1 # add to coordinate part
                #for ivs in range(self.ivs+1):
                 #   self.dataTrans[i,ibttm:,ivs]=scipy.interp(self.zi[i,ibttm:],depths[0,ibttm:],lvldata[0,ibttm:,ivs])*w[i,0] + \
                  #  scipy.interp(self.zi[i,ibttm:],depths[1,ibttm:],lvldata[1,ibttm:,ivs])*w[i,1] + \
                   # scipy.interp(self.zi[i,ibttm:],depths[2,ibttm:],lvldata[2,ibttm:,ivs])*w[i,2]	
                    #self.dataTrans[i,-1,ivs]=(lvldata[:,-1,ivs]*w[i,:]).sum() 
                    #self.dataTrans[i,ibttm,ivs]=(lvldata[:,ibttm,ivs]*w[i,:]).sum() 
				#ivs at beginning	
                #for ivs in range(self.ivs+1):
                 #   self.dataTrans[ivs,i,ibttm:]=scipy.interp(self.zi[i,ibttm:],depths[0,ibttm:],lvldata[ivs,0,ibttm:])*w[i,0] + \
                  #  scipy.interp(self.zi[i,ibttm:],depths[1,ibttm:],lvldata[ivs,1,ibttm:])*w[i,1] + \
                   # scipy.interp(self.zi[i,ibttm:],depths[2,ibttm:],lvldata[ivs,2,ibttm:])*w[i,2]	
                    #self.dataTrans[ivs,i,-1]=(lvldata[ivs,:,-1]*w[i,:]).sum() 
                    #elf.dataTrans[ivs,i,ibttm]=(lvldata[ivs,:,ibttm]*w[i,:]).sum() 					
                #self.dataTrans=np.ma.masked_array(self.dataTrans,mask=self.dataTrans==0.0)

            print('usig nearest neighbour interpolation for transect')
            d,qloc=self.xy_nn_tree.query(self.coords)
            isub=np.unique(qloc,return_index=True)[1] # only use unique values
            qloc=np.asarray([indi for indi in qloc if indi in qloc[isub]])
            #self.dataTrans=data[:,qloc[isub],:]
            self.dataTrans=data[:,qloc,:]
            #self.dataTrans=np.ma.masked_array(self.dataTrans,mask=np.tile(vert[qloc[isub]].mask,(3,1,1)))
            self.dataTrans=np.ma.masked_array(self.dataTrans,mask=np.tile(vert[qloc].mask,(3,1,1)))
            #self.zi=zcor[qloc[isub],:]
            self.zi=zcor[qloc,:]
            for i in range(self.zi.shape[0]):				                
                #self.zi[i,:self.ibbtm[qloc[isub][i]]]=self.zi[i,self.ibbtm[qloc[isub][i]]]
                self.zi[i,:self.ibbtm[qloc[i]]]=self.zi[i,self.ibbtm[qloc[i]]]
            self.zi.mask=False
            #self.xs=self.xs[:len(isub),:]				
            self.xs=self.xs[:len(qloc),:]				
		
            comps=[' - u', '- v ', '- abs' ]
            ax3=plt.subplot(3,1,3)
            ph,ch=self.plot_transect(np.sqrt(self.dataTrans[0,:,:]**2+self.dataTrans[1,:,:]**2),is2d)
            ch.set_label(self.varname + comps[-1])			
            if  not is2d:		
                plt.clim(self.clim) 

            if self.newio<2:
                plt.title(str(self.reftime + dt.timedelta(seconds=int(self.ncs[self.filetag]['time'][self.total_time_index]))))
            else:
                plt.title(str(self.ncs[self.filetag]['time'][self.total_time_index]))
				
            for iplt in range(0,2):
                plt.subplot(3,1,iplt+1,sharex=ax3,sharey=ax3)
                ph,ch=self.plot_transect(self.dataTrans[iplt,:,:],is2d)
                ph.set_cmap(cmo.balance)
				
                if self.quivVar.get():
                    plt.quiver(self.xs,self.zi,self.dataTrans[0,:,:],self.dataTrans[-1,:,:],color='w')
                plt.tick_params(axis='x',labelbottom='off')
                ch.set_label(self.varname + comps[iplt])
                plt.gca().set_ylim(ylim)
                #plt.clim(self.clim) 
                plt.xlim((self.xs.min(),self.xs.max()))	


						
        plt.gca().set_ylim(ylim)
        #if  not is2d:		
        #    plt.clim(self.clim) 
        plt.xlabel('transect length [points]')
        plt.xlim((self.xs.min(),self.xs.max()))			 #insert extra
		
        plt.tight_layout()
        #self.update_plots()		
		
        self.zminfield.delete(0, 'end'),self.zmaxfield.delete(0, 'end')
        self.zminfield.insert(8,str(plt.gca().get_ylim()[0])),self.zmaxfield.insert(8,str(plt.gca().get_ylim()[1]))
		#self.zminfield.insert(8,str(ylim[0])),self.zmaxfield.insert(8,str(ylim[1]))
		#self.zminfield.insert(8,str(ylim[0])),self.zmaxfield.insert(8,(ylim[1]))
        #self.update_plots()
		
    # navigate time steps    
    def firstts(self):
        self.stack.set(self.stacks[0])
    def prevstep(self):
        if self.ti>=0:
            self.ti_tk.set(self.ti-1)
        elif self.stack.get()-1 in self.stacks:
            self.stack.set(self.stack.get()-1)
    def nextstep(self):
        if self.ti<self.stack_size-1:
            self.ti_tk.set(self.ti+1)
        elif self.stack.get()+1 in self.stacks:
            self.stack.set(self.stack.get()+1)
        else:   
            self.stack.set(self.stacks[0])
    def lastts(self):
        self.stack.set(self.stacks[self.nstacks-1])
        self.ti_tk.set(self.stack_size-1)

    def updateaxlim(self,*args):
        plt.figure(self.fig0)
        axes = plt.gca()
        xmin,xmax=self.xminfield.get(),self.xmaxfield.get()
        ymin,ymax=self.yminfield.get(),self.ymaxfield.get()
        if (len(xmax)*len(ymax)*len(ymin)*len(ymax))>0:
            axes.set_xlim([np.double(xmin),np.double(xmax)])
            axes.set_ylim([np.double(ymin),np.double(ymax)])
            self.update_plots()
            
    def updatezlim(self,*args):
        #if self.fig1 in plt.get_fignums() and (self.extract!=self.timeseries):
        if self.activefig in plt.get_fignums() and (self.extract!=self.timeseries):
            zmin,zmax=self.zminfield.get(),self.zmaxfield.get()
            if (len(zmax)*len(zmin))>0 : 
                #cf=plt.figure(self.fig1)
                cf=plt.figure(self.activefig)
                if self.ivs==1:
                    axes = plt.gca()
                    cf.get_axes()
                    axes.set_ylim([np.double(zmin),np.double(zmax)])
                else:
                    if self.extract==self.profiles:
                        m,n=1,3
                    else:
                        m,n=3,1
						
                    #axes = plt.gca()
                    ax=cf.get_axes()
                    for axi in ax:			
                        if not 'colorbar' in axi.get_label():	
                            axi.set_ylim([np.double(zmin),np.double(zmax)])
                    #for i in range(1,4):
                        #plt.subplot(m,n,i)    
                    #    axes = plt.gca()
                    #    cf.get_axes()
                    #    axes
                self.update_plots()
                
    def savenodevalues(self,*args):
        outname=filedialog.asksaveasfilename(filetypes = (("text files ","*.txt"),("all files","*.*"),("nc files","*.nc")))
        ext=outname[outname.index('.')+1:]
        if ext!='nc':
            np.savetxt(outname,self.nodevalues)
        else:
            ncout = Dataset('test2.nc', 'w', format='NETCDF4')
            ncout.setncattr('history',  str(dt.date.today()) + 'schism_view2.py layer extract of ncout file')
            #ncout.createDimension('nSCHISM_vgrid_layers', 1)
            ncout.createDimension('time', 1)
            ncout.createDimension('nSCHISM_hgrid_node', len(self.nodevalues))
            ncout.createDimension('nMaxSCHISM_hgrid_face_nodes', 4)
            ncout.createDimension('nSCHISM_hgrid_face',len(self.nc.dimensions['nSCHISM_hgrid_face']))
            
            # variables
            time = ncout.createVariable('time', 'f8', ('time'))
            time[:]=self.nc['time'][self.total_time_index]  
            time.setncatts(self.nc['time'].__dict__)
            
            varout = ncout.createVariable('SCHISM_hgrid_node_x', 'f8', ('time','nSCHISM_hgrid_node'))                
            varout[:]=self.nc['SCHISM_hgrid_node_x'][:]
            varout.setncatts(self.nc['SCHISM_hgrid_node_x'].__dict__)      
            
            varout = ncout.createVariable('SCHISM_hgrid_node_y', 'f8', ('time','nSCHISM_hgrid_node'))                
            varout[:]=self.nc['SCHISM_hgrid_node_y'][:]
            varout.setncatts(self.nc['SCHISM_hgrid_node_y'].__dict__)      
            
            varout = ncout.createVariable('SCHISM_hgrid_face_nodes', 'i', ('nSCHISM_hgrid_face','nMaxSCHISM_hgrid_face_nodes'))                
            varout.setncatts(self.nc['SCHISM_hgrid_face_nodes'].__dict__)      
            varout[:]=self.nc['SCHISM_hgrid_face_nodes'][:] #.filled((nc['SCHISM_hgrid_face_nodes']._FillValue))
            ncout.close()

            #varout = ncout.createVariable(varname, 'f8', ('time','nSCHISM_hgrid_node'))                
            #varout[:]=nodevalues
            #ncout.close()

    def savecoords(self,*args):
            outname=filedialog.asksaveasfilename(filetypes = (("bp files ","*.bp"),("all files","*.*")))
            f=open(outname,'w')
            f.write(outname+'\n')
            f.write('{:d}\n'.format(len(self.coords)))
            mat=np.zeros((len(self.coords),4))
            mat[:,0]=1+np.arange(len(self.coords))
            mat[:,-1]=np.ones(len(self.coords))
            mat[:,1:3]=np.asarray(self.coords)
            np.savetxt(f,mat,fmt='%i %.6f %.6f %i')
            f.close()
            
    def savets(self,*args):
            outname=filedialog.asksaveasfilename(filetypes = (("dat files ","*.dat"),("all files","*.*")))
            header='timeseries extract of variable '+ self.varname + ' \n from ncdata ' + self.combinedDir + '\n time \ ts at (next neighbours) coordinates '+str(self.coords) 
            mat=np.column_stack((self.t,self.ts))
            np.savetxt(outname,mat,header=header)
            
    def savehov(self,*args):
            outname=filedialog.asksaveasfilename(filetypes = (("dat files ","*.dat"),("all files","*.*")))
            name0=outname[:outname.index('.')]
            name1=outname[outname.index('.'):]
            outname1=name0+'_hov_zcor'+name1
            outname2=name0+'_hov_data'+name1
            header='hovmoeller extract of variable '+ self.varname + ' \n from ncdata ' + self.combinedDir + '\n time \ ts at (next neighbours) coordinates '+str(self.coords) 
            mat=np.column_stack((self.t,self.ts))
            mat[mat==self.ts.mask]=np.nan
            np.savetxt(outname1,mat,header=header)
            header='hovmoeller extract of variable zcor \n from ncdata ' + self.combinedDir + '\n time \ ts at (next neighbours) coordinates '+str(self.coords) 
            mat=np.column_stack((self.t,self.zcor))
            mat[mat==self.ts.mask]=np.nan
            np.savetxt(outname2,mat,header=header)
            print('done writing hovmoeller')
			
    def comp_obs(self,*args):
            ''' plot against observation file '''	
            obsfile=filedialog.askopenfile(title='select observation file',filetypes = (('bp files', '*.nc'),('All files', '*.*'))).name
            self.dates=np.asarray([self.reftime + dt.timedelta(seconds=ti) for ti in            self.ncs[self.filetag]['time'].values],np.datetime64)
            if self.newio<2:
                self.dates=np.asarray([self.reftime + dt.timedelta(seconds=ti) for ti in            self.ncs[self.filetag]['time'].values],np.datetime64)	            
            else:
                self.dates=self.ncs[self.filetag]['time'].values       
			
            self.obsnc=xr.open_dataset(obsfile)
            self.obsnc=self.obsnc.sel(TIME=slice(self.dates[0],self.dates[-1]))
            
            lonq=np.unique(self.obsnc['LONGITUDE'])[0]
            latq=np.unique(self.obsnc['LATITUDE'])[0]
            self.d,self.nn=self.ll_nn_tree.query((np.vstack([lonq.ravel(), latq.ravel()])).transpose())
            #self.coords=list((self.x[ self.nn[0]],self.y[ self.nn[0]]))
            self.coords=(self.x[ self.nn[0]],self.y[ self.nn[0]])
            self.npt=len(self.nn)
            if self.npt==1:
                self.coords=[self.coords]
            self.timeseries(coords=self.coords)
            
            ph2=self.obsnc['TEMP'].plot()
            plt.legend(['schism ' + self.varname,obsfile[obsfile.rindex('/'):]])
            self.update_plots()				
			
    def client_exit(self):
            plt.close('all')
            self.master.destroy()
            exit()

# launch gui
root = tk.Tk()
root.geometry("420x460")
#root.geometry("420x480")
root.grid_rowconfigure(12, minsize=100)  
root.grid_columnconfigure(4, minsize=100)  
if __name__ == "__main__":
	app= Window(root)
	root.title('schout_view')
	root.mainloop()
