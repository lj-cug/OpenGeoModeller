 PROGRAM MAIN 
   implicit none
   integer,parameter :: nmax = 5
   integer :: i,j
   real:: aa, bb, cc
   real,dimension(nmax) :: a,b,c
   ! 
   ! The first fortran program 
   ! 
   aa = 5.0 
   bb = 8.0 
   
   call add(aa,bb,cc)  ! Fortran中使用数值传递
   
   do i=1,nmax
      a(i) = i
      b(i) = i+1 
   enddo
   
   call arrayadd(nmax,a,b,c)
   
   pause
  end program MAIN 
