      program main
	implicit none
	character fileread*50, filewrite*50  !读取文件名
	character DataName*50
	real*8 :: xllcorner,yllcorner
	real*8,allocatable:: zval(:) !地形高程值
	real*8 :: Xcoord,Ycoord
	real*8 cellsize
	integer*4 i,j,k,ncols,nrows,NODATA_value,counter
    
	print *, 'Input filename:'
	read(*,*) fileread
	
	!fileread=fileread//'.asc'         !...读取DEM的ASCII数据文件
	open(unit=2,file=fileread,status='old')
	
      filewrite='xyz.dat'         !...形成的河道地形数据文件
	open(unit=3,file=filewrite)
      write(*,*) 'reading DEM data...'
      read(2,*) DataName,ncols
	read(2,*) DataName,nrows
	read(2,*) DataName,xllcorner
	read(2,*) DataName,yllcorner
	read(2,*) DataName,cellsize
	read(2,*) DataName,NODATA_value
      
      write(*,*) 'writing TECPLOT data...'
      yllcorner = yllcorner + (nrows-1)*cellsize
	allocate (zval(ncols))
      
	write(3,*) 'variables = x,y,z'  ! for Tecplot IDW interpolation	
	counter = 0
      do i=1,nrows
	          Ycoord=yllcorner-cellsize*(i-1)
	        read(2,*)(zval(k),k=1,ncols)
		    do j=1, ncols
              Xcoord=xllcorner+cellsize*(j-1)
	          if (zval(j)>-1000.) then    ! 踢出无效数据
	              counter = counter + 1
	              if(mod(counter, 10)==0) write(3,10) Xcoord,Ycoord,zval(j)
	          endif
	        enddo
	enddo
10    format(f16.3,1x,f16.3,1x,f10.3)
	close(2)
	close(3)
	
	write(*,*) 'Total extracted point number:',counter
	print *, 'Elev are extracted succefully!'
	stop
	end