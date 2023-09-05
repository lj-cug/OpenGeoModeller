	program  main
    implicit none
	integer :: i,j,k,l	
	character(len=15) :: filename1,filename2
	real,allocatable,dimension(:) :: X,Y,ZB
	integer,allocatable,dimension(:,:) :: nep
    integer :: nelem, nodes
    
    print *, 'The input gr3 filename:'
    read(*,*) filename1
	OPEN(1,FILE=filename1,status='old')  
	
    print *, 'The output tecplot filename:'
    read(*,*) filename2
	OPEN(2,FILE=filename2,status='replace')

	read(1,*) 
	read(1,*) nelem,nodes
	
	!
	allocate(x(nodes),y(nodes),zb(nodes),nep(nelem,3))

	DO I=1,NODES
	  READ(1,*)j,X(I),Y(I),ZB(I)    
	ENDDO

	DO J=1,NELEM
	   READ(1,*)k,l,(NEP(J,1:3))
	ENDDO
	
! Write Tecplot header
      write(2,*) 'title="Mesh"'
      write(2,*)'variables= "x","y","z"'
      write(2,*)'zone T="ZONE 001" '
      write(2,90)'Nodes=',nodes,',Elements=',nelem,  &
         & ', ZONETYPE=FETriangle'
      write(2,*)'DATAPACKING=POINT'
90  format(a6,1x,I6,1x,a10,1x,I6,1x,a25)

    DO I=1,NODES
	  write(2,100)X(I),Y(I),ZB(I)    
	ENDDO

	DO J=1,NELEM
	   Write(2,200)(NEP(J,1:3))
	ENDDO
	
100 format(3(1x,f12.6))
200	format(3(1x,I6))
	write(*,*)"data change complete normally!"
	stop
	END
