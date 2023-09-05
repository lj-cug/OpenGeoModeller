program main
	implicit none
	character(50) :: filename1,filename2
	character(100) :: dataTemp
	integer :: i,j,k,numnodes,numeles,numelep1,numelep2
!	
	integer,allocatable,dimension(:) :: nEle,nNode
	integer,allocatable,dimension(:,:):: nNodes
	real(kind=8),allocatable,dimension(:) :: x,y,z
	real(kind=8),allocatable,dimension(:,:):: Node_x

!Gambit导出的网格文件  Polyflow求解器格式
!	filename='confluence.neu'
	write(*,*)'输入neu文件：*.neu'
	!read(*,*) filename1
	filename1 = 'pier.neu'
	open(unit=1,file=filename1,status='old')

	write(*,*)'输出GMSH文件：*.msh'
	!read(*,*)filename2
	filename2 = 'pier.msh'
	
!---------- Read from GAMBIT neutral file ----------------------
      do k=1,9
	  if (k.ne.7) then
          read(1,*) !dataTemp
	  else
          read(1,*) numNodes, numEles  !,temp1,temp2,temp3,temp4
	    allocate(x(numNodes),y(numNodes),z(numNodes),nNode(numNodes),  &
		         nNodes(3,numEles),nEle(numEles))   	     
	  endif
	enddo

    do i=1, numNodes
	  read (1,*) nNode(i), x(nNode(i)), y(nNode(i)) !,z(nNode(i))
	enddo
	do k=1,2
	  read(1,*) dataTemp
	enddo
	do i=1, numEles
	   read (1,*) nEle(i),NumElep1,numElep2,(nNodes(k,nEle(i)),k=1,numElep2) 
    enddo
    close(1)

! Write to GMSH 2D file
  allocate(Node_x(2,numNodes))
  do i=1,numNodes
     Node_x(1,i)= x(i)
     Node_x(2,i)= y(i)  
  enddo
  
  call gmsh_mesh2d_write ( filename2, 2, numNodes, Node_x, 3, numEles, nNodes )
   
!
  deallocate(x,y,Node_x,z,nNode,nNodes,nEle)
  stop
end
	