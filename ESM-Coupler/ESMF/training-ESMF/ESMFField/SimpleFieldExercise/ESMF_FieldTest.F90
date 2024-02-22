!
! Demonstrates how to create an ESMF_Grid and two ESMF_Fields.
!
! Should be run with 4 PETs:
!
! $ mpirun -np 4 ./ESMF_FieldTest
!
program ESMF_FieldTest

    use ESMF

    implicit none

    ! local variables
    type(ESMF_Grid)             :: grid3d
    type(ESMF_Field)            :: field3d
    type(ESMF_Field)            :: field3d2
    real(ESMF_KIND_R8), pointer :: farrayPtr3D(:,:,:)

    type(ESMF_VM) :: vm
    integer :: localPet
    integer :: rc

    integer :: totalLBound(3)
    integer :: totalUBound(3)
    integer :: totalCount(3)

    integer :: xdim, ydim, zdim

    call ESMF_Initialize(rc=rc)
    if (ESMF_LogFoundError(rcToCheck=rc)) &
      call ESMF_Finalize(endflag=ESMF_END_ABORT)

    call ESMF_VMGetCurrent(vm, rc=rc)
    call ESMF_VMGet(vm, localPet=localPet, rc=rc)

    xdim = 180
    ydim = 90
    zdim = 50

    ! 
    ! FIX ME
    !
    ! Create an ESMF_Grid 180 x 90 x 50 cells
    ! Decomposition 2 x 2 x 1
    ! Non-periodic
    ! Assign it to the variable grid3d
    
    
    
    



    
    ! get distgrid for center stagger
    !call ESMF_GridGet(grid=grid3d, staggerloc=ESMF_STAGGERLOC_CENTER, &
    !       distgrid=distgrid3d, rc=rc)
    !if (ESMF_LogFoundError(rcToCheck=rc)) &
    !  call ESMF_Finalize(endflag=ESMF_END_ABORT)

    ! get total cells in each dimension for center stagger
    call ESMF_GridGetFieldBounds(grid=grid3d, localDe=0, &
        staggerloc=ESMF_STAGGERLOC_CENTER, totalCount=totalCount, rc=rc)
    if (ESMF_LogFoundError(rcToCheck=rc)) &
      call ESMF_Finalize(endflag=ESMF_END_ABORT)

    allocate(farrayPtr3D(totalCount(1), totalCount(2), totalCount(3)))

    !print *, "PET ", localPet, " totalCount = ", totalCount(:)

    ! create Field from Fortran array pointer
    field3d = ESMF_FieldCreate(grid3d, farrayPtr3D, &
        indexflag=ESMF_INDEX_DELOCAL, rc=rc)
    if (ESMF_LogFoundError(rcToCheck=rc)) &
      call ESMF_Finalize(endflag=ESMF_END_ABORT)


    ! FIX ME

    ! Create another field (on same grid) by providing typekind of ESMF_TYPEKIND_R8
    ! ESMF allocated memory
    ! Field is at center stagger location
    ! Field should have halo padding of 2 in all directions
    ! Assign to variable field3d2
    






    call ESMF_FieldGetBounds(field3d2, &
      totalLBound=totalLBound, &
      totalUBound=totalUBound, &
      rc=rc)
    if (ESMF_LogFoundError(rcToCheck=rc)) &
      call ESMF_Finalize(endflag=ESMF_END_ABORT)

    print *, "PET ", localPet, " totalLBounds = ", totalLBound(:)
    print *, "PET ", localPet, " totalUBounds = ", totalUBound(:)


    call ESMF_FieldDestroy(field3d)
    call ESMF_FieldDestroy(field3d2)
    call ESMF_GridDestroy(grid3d)

    deallocate(farrayPtr3D)

    call ESMF_Finalize()


end program

