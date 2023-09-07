    program main
	implicit none
	character fileread*50, filewrite*50  
	character DataName*50	
	integer :: i,j,k,ncols,nrows,NODATA_value,count	
	real :: xllcorner,yllcorner
	real :: Xcoord,Ycoord
	real :: cellsize	
	integer,allocatable:: zval(:)         !  1 for stream Line
	
	fileread='stream_xxr.asc'      
	open(unit=2,file=fileread,status='old')
	
    filewrite='channel.dat' 
	open(unit=3,file=filewrite,status='replace')
	
	open(unit=1,file='pts.m',status='replace')

    read(2,*) DataName,ncols
	read(2,*) DataName,nrows
	read(2,*) DataName,xllcorner   ! low left corner
	read(2,*) DataName,yllcorner
	read(2,*) DataName,cellsize
	read(2,*) DataName,NODATA_value
	
    write(*,*) xllcorner,yllcorner

    yllcorner = yllcorner + (nrows-1)*cellsize   ! move to high left corner
	allocate (zval(ncols))
      
	write(3,*) 'variables = x,y,z'
	write(1,*)'pts{1,1}=['  
      do i=1,nrows
            count = 0
	        Ycoord=yllcorner-cellsize*(i-1)
	        read(2,*)(zval(k),k=1,ncols)
		    do j=1, ncols
              Xcoord=xllcorner+cellsize*(j-1)
	          if (zval(j) == 1) then 
		          count = count + 1	            
	              write(3,10) Xcoord,Ycoord,zval(j)
	              write(1,'(2(f15.2,1x),a1)')Xcoord,Ycoord,';'	              
	          endif
	        enddo
!	        
	  if(count > 0) then
	      write(*,*)'row:',i,'Number of point:',count
	  endif     	        
	enddo
	write(1,*)']'
10 format(2(f15.3,1x),I2)
    close(1)
	close(2)
	close(3)
	print *, 'Stream lines are taken completely!'
    stop
	end