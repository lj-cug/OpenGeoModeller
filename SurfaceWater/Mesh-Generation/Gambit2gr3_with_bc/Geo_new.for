	program geo
      parameter(iopnum=2,landnum=1) !开边界和陆地边界数

	integer ii,NUMNP,NELEM,NGRPS,NBSETS,NDFCD,NDFVL	
	integer stat
	integer itotalopnodenum
	real,allocatable :: fNodeX(:),fNodeY(:)
	integer,allocatable ::iNode(:,:)
	integer,allocatable :: iopnodenum(:),lbnodenum(:)

	dimension iopnode(iopnum,1000),lbnode(landnum,1000) 
	              !维数=边界数*边界上节点数
!==========================information of node from Gambit================
	open(1,file='ch.neu',status='old')
	do i=1,6
		read(1,*)
	end do
          read(1,*) NUMNP,NELEM,NGRPS,NBSETS,NDFCD,NDFVL	
  		read(1,*)
		read(1,*) 
		
	allocate(fNodeX(NUMNP),fNodeY(NUMNP),iNode(NELEM,4),stat=stat)		
      if(stat/=0) write(*,*) 'array allocated fail.'

	allocate(iopnodenum(iopnum),lbnodenum(landnum),stat=stat)
	if(stat/=0) write(*,*) 'array allocated fail.'
			      
	do i=1,NUMNP
		read(1,*)ii,fNodeX(i),fNodeY(i)
	fNodeX(i)=1*fNodeX(i)
	fNodeY(i)=1*fNodeY(i)
	end do

	read(1,*)
	read(1,*)
	do i=1,NELEM
		read(1,*)ii,id1,id2,(iNode(i,j),j=1,id2) 
		if(id2==3)iNode(i,4)=iNode(i,1)             
	end do

!======================open boundary search===============================
	
	do k=1,iopnum
	 itotalopnodenum=0
		read(1,*)obname,fluentype0,iopnodenum(k)		 
		read(1,*)(iopnode(k,i),i=1,iopnodenum(k))
		itotalopnodenum=itotalopnodenum+iopnodenum(k)
	end do

!======================land boundary search===============================
	
	do k=1,landnum
	  itotalopnodenum=0
		read(1,*)lbname,fluentype1,lbnodenum(k)		 
		read(1,*)(lbnode(k,i),i=1,lbnodenum(k))
		itotalopnodenum=itotalopnodenum+lbnodenum(k)
	end do

	read(1,*)
	CLOSE(1)

!=====================write hgrid.gr3=========================================
	open(2,file='hgrid.gr3',status='unknown')
	write(2,*) 'hgrid.gr3'
	write(2,*) NUMNP,NELEM

	do i=1,NUMNP
		write(2,5)i,fNodeX(i),fNodeY(i)
	end do
5     format(I10,2(1x,f12.3))

	do i=1,NELEM
		if(iNode(i,4)==iNode(i,1))then 
			id=3
			write(2,*)i,id,(iNode(i,j),j=1,id)
		else 
			id=4
		write(2,*)i,id,(iNode(i,j),j=1,id)
	endif
	end do

	do k=1,iopnum
	if(k==1)then
		write(2,10)iopnum,"= Number of open boundaries"
	    write(2,10)itotalopnodenum,"= Total number of open boundary nodes"
      endif

	  write(2,10)iopnodenum(k),"= Number of nodes for open boundary",k
 
		do i=1,iopnodenum(k)
			write(2,20)iopnode(k,i)
		end do
	end do

	do k=1,landnum
	if(k==1)then
	    write(2,10)landnum,"= Number of land boundaries"
		write(2,10)itotalopnodenum,"= Total number of land boundary nodes"
      endif

		write(2,10)lbnodenum(k),"= Number of nodes for open boundary",k

		do i=1,lbnodenum(k)
		  write(2,20)lbnode(k,i)
		end do
	end do
10    format(I10,2x,A40,1x,I5)
20    format(I10)
	close(2)

      deallocate(fNodeX,fNodeY,iNode,stat=stat)
      if(stat/=0) write(*,*) 'array deallocated fail.'
      deallocate(iopnodenum,lbnodenum,stat=stat)
      if(stat/=0) write(*,*) 'array deallocated fail.'

	stop
	end