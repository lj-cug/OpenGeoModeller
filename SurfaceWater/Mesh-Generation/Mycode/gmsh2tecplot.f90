  program main
	implicit none
	character :: filename*50
	character :: dataTemp*100
	integer :: temp1,temp2,temp3,temp4,temp5
	integer :: m,numnodes,numeles
	integer :: i,j,k,mm
    real :: temp
	real*8,allocatable:: x(:),y(:),z(:)
	integer,allocatable::nNodes(:,:)


!Gambit导出的网格文件  Polyflow求解器格式
!	filename='confluence.neu'
	write(*,*)'输入MSH文件：*.msh'
	!read(*,*) filename
    filename = 'hth.msh'
	open(unit=2,file=filename,status='old')

	write(*,*)'输出Tecplot文件：*.dat'
	!read(*,*)filename
    filename='hth.dat'
	open(unit=3,file=filename,status='replace')
	
!---------- Read from GMSH file ----------------------
      do m=1,4
        read(2,*)    ! headers
      enddo
      read(2,*)numNodes
      allocate(x(numNodes),y(numNodes),z(numNodes))
      do i=1,numNodes
         read(2,*)temp1,x(i),y(i),z(i) 
      enddo
      write(*,*)'读取节点坐标及地形完毕.'
      read(2,*)
      read(2,*)
      read(2,*)numEles
      allocate(nNodes(3,numEles))
	  do i=1, numEles
	     read (2,*) temp1,temp2,temp3,temp4,temp5,(nNodes(k,i),k=1,3)
      enddo
      write(*,*)'读取单元节点编号完毕.'

!--------- TO Tecplot -------------------------  
	write(*,*) 'Nodes:',numNodes, 'Elements:',numEles
    write(3,*) 'TITLE     = ""  '
	write(3,*) 'VARIABLES = "x"'
	write(3,*) '"y"'
	write(3,*) '"z"'
! trianglar element
    write(3,*)'ZONE T="triangle"'
	write(3,*) 'N=',numNodes,',E=',numEles,',ZONETYPE=FETRIANGLE'
	write(3,*) 'DATAPACKING=POINT'
	write(3,*) 'DT=(SINGLE SINGLE SINGLE )'
	do i=1, numNodes
        write(3,101) x(i), y(i), z(i)
	enddo
	do 90 i=1, numEles
        write(3,102)(nNodes(k,i),k=1,3)      
   90 enddo

  101 format(3(f15.5,1x))
102 format(3(I8,1x))      

    
!----------TO Gr3 format file----------------------
 open(unit=1,file='hth.grd',status='replace')
 	write(1,*) 
	write(1,*) numEles,numNodes

	DO I=1,numNodes
	  write(1,103)I,X(I),Y(I),Z(I)    
	ENDDO

	DO J=1,numEles
	   write(1,104)J,3,(nNodes(k,i),k=1,3) 
	ENDDO    

      close(1)
	  close(2)	  
	  close(3)     
103 format(I8,1x,3(f15.5,1x))
104 format(I8,1x,I1,1x,3(I8,1x))  
      print *, 'ok!'
	  
	  stop
	end