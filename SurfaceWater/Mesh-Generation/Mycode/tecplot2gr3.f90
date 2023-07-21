program main
implicit none
integer :: i,j,k
integer :: nelem,node
character :: elech,nodech,filename*50

real,allocatable,dimension(:) :: x,y,h
integer,allocatable,dimension(:,:) :: elnode

write(*,*)'Input Tecplot data file: [*.dat]'
read(*,*) filename
open(1,file=filename,status='old')

write(*,*)'Input the elnode and node number: (nelem,node)'
read(*,*) nelem,node
allocate(x(node),y(node),h(node),elnode(nelem,3))

open(2,file='hgrid.gr3',status='replace')

do i=1,8
  read(1,*)
enddo
 
write(2,*) 'hgrid.gr3'
write(2,*)  nelem,node

write(*,*) 'reading and writing node'
do i=1,node
     read(1,*) x(i),y(i),h(i)
 !    write(2,10) i,x(i),y(i),h(i)
     write(2,10) i,x(i),y(i),0-h(i)   ! for Telemac2D,要把水深翻转一下
enddo

write(*,*) 'reading and writing element'
do i=1,nelem
     read(1,*) (elnode(i,j),j=1,3)
     write(2,11) i,3,(elnode(i,j),j=1,3)   
enddo

10 format(I6,1x,f20.8,1x,f20.8,1x,f10.3)
11 format(I6,1x,I1,1x,3(I6,1x))
stop
end