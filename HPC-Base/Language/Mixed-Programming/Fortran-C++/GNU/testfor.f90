module testfor
   use iso_c_binding
   implicit none
   
contains

    subroutine testingfor(x,string,l) bind(c, name="testingfor")
        implicit none
        integer,intent(in) :: l
        double precision, intent(in) :: x
        character,dimension(*), intent(in) :: string    ! len=50 error
! local variables
        integer :: i,j
    
        print *, "# calling testfor in Fortran(Char)" 
        do i=1,l      
          write(*,*)string(i)
        enddo
        print *, "# calling testfor in Fortran(Number)"
        print *, "x = ", x 
    end subroutine
    
end module testfor



module test

  implicit none

  integer, private :: lng = 2
  integer, private :: lu = 6
  character(len=150), dimension(2) :: message_1
  character(len=150), dimension(2) :: message_2 
  
  
   interface writing
      module procedure write_str
   end interface
   
   contains
   
subroutine write_str(str)
!-----------------------------------------------------------------------
! Description of the subroutine :
! Writes a string to the listing.
!-----------------------------------------------------------------------
  implicit none

! integer :: lng,lu
! common/info/lng,lu
!-----------------------------------------------------------------------
! Arguments
      character(len=*), dimension(2), intent(in) :: str
!-----------------------------------------------------------------------
! Print the message to the listing
  select case (lng)
     case (2)
       write(*,*) trim(str(1))
     case (1)
       write(*,*) trim(str(2))
  end select
!-----------------------------------------------------------------------
end subroutine write_str
end module test