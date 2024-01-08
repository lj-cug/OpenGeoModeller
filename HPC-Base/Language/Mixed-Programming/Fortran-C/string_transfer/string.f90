!
module string
implicit none
interface

subroutine print_c(string) bind(c)
use iso_c_binding, only: c_ptr, c_char
  character(kind=c_char), dimension(*) :: string
end subroutine

!
end interface
!
end module string


program main
use iso_c_binding
use string

implicit none
integer :: i
character(len=50) :: test1

test1 = "Test to print FORTRAN string in C program."

call print_c(test1)

write(*,*) "OK!"
end