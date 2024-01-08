program vector
!
!  演示Fortran将vector传递到C程序中
!
implicit none

interface
  subroutine vec(r,len) bind(c)
    use,intrinsic :: iso_c_binding
    implicit none
    integer(c_int) :: len
    real(c_double) :: r(0:len)
   end subroutine vec
end interface

double precision,allocatable :: r(:)
integer :: len,i

len=5
allocate(r(0:len))

do i=0,len
   r(i) = i*0.1
enddo

print *, "Fortran calling C, passing array"

call vec(r,%val(len))

stop

end program vector