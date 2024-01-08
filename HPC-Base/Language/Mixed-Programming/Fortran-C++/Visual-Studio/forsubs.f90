! forsubs.f90
!
! FUNCTIONS/SUBROUTINES exported from FORSUBS.dll:
! FORSUBS - subroutine
!
INTEGER*4 FUNCTION Fact (n)
!DEC$ ATTRIBUTES DLLEXPORT::Fact
INTEGER*4 n [VALUE]
INTEGER*4 i, amt

amt = 1
DO i = 1, n
	amt = amt * i
END DO
Fact = amt
write(*,*)"Mixed calls succeed!"
END

SUBROUTINE Pythagoras (a, b, c)
!DEC$ ATTRIBUTES DLLEXPORT::Pythagoras
REAL*4 a [VALUE]
REAL*4 b [VALUE]
REAL*4 c [REFERENCE]

c = SQRT (a * a + b * b)

END