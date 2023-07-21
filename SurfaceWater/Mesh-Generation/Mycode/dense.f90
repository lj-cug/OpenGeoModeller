program main
    
implicit none

integer :: i,j,np
real*8,dimension(300) :: x,y,x1,y1
    
open(unit=1,file='xy_old.dat',status='old')    
open(unit=2,file='xy_new.dat',status='replace')    

np=114  ! xy 坐标点的个数
do i=1,np
    read(1,*)x(i),y(i) 
enddo

do i=1,np-1
    x1(i)=(x(i)+x(i+1))/2.0
    y1(i)=(y(i)+y(i+1))/2.0 
enddo
x1(np)=x(np);y1(np)=y(np)

do i=1,np
   write(2,*)x(i),y(i) 
   write(2,*)x1(i),y1(i) 
enddo



close(1)
close(2)
stop
end