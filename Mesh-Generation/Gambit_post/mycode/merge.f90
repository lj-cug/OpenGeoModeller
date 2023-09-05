	program  main
    implicit none
	integer :: i,j,k,l	
	character(len=15) :: filename1,filename2
	real*8,allocatable,dimension(:) :: X,Y,ZB,zb2
	integer,allocatable,dimension(:,:) :: elnode
    integer :: nelem, nodes
    real*8 :: t1,t2
    
    print *, 'The new gr3 filename:'
 !   read(*,*) filename1
 	filename1 = 'hgrid_new.gr3'
	OPEN(1,FILE=filename1,status='old')  

    print *, 'The new tecplot filename:'
  !  read(*,*) filename2
  	filename2 = 'mesh_topo.dat'
	OPEN(2,FILE=filename2,status='old')

	read(1,*) 
	read(1,*) nelem,nodes	
	!
	allocate(x(nodes),y(nodes),zb(nodes),zb2(nodes),elnode(nelem,3))

	DO I=1,NODES
	!  READ(1,*)j,x(i),y(i),ZB(I)    
	ENDDO

!	DO i=1,NELEM
!	   READ(1,*)k,l,(elnode(i,j),j=1,3)
 !   ENDDO
    
       
! ¶ÁÈ¡tecplotÊý¾Ý
do i=1,9
   read(2,*)  ! header
enddo

do i=1,nodes
   read(2,*) x(i),y(i),zb2(i)
enddo 
	DO i=1,NELEM
	   READ(2,*)(elnode(i,j),j=1,3)
    ENDDO

    close(1)
    close(2)
    
   open(3,file='hgrid_new2.gr3',status='replace')
   write(3,*)"Mesh_Topo"
   write(3,*) nelem,nodes	
   
 	DO I=1,NODES
	   write(3,10)i,X(I),Y(I),ZB2(I)    
    ENDDO  
10  format(I6,1x,f15.6,1x,f15.6,1x,f10.3)
    
  	DO i=1,NELEM
	   write(3,20)i,3,(elnode(i,j),j=1,3)
    ENDDO 
20 format(I6,1x,I1,1x,3(I8,1x))    
close(3)
   
   deallocate(x,y,zb,zb2,elnode)
    
    stop
    end