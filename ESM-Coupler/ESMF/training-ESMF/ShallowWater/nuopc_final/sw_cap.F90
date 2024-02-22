module sw_cap
	
    use ESMF
    use NUOPC
    use NUOPC_Model, model_SetServices => SetServices, &
        model_label_Advance => label_Advance

    use sw_refactor, only: &
        sw_nx => nx, &
        sw_ny => ny, &
        sw_dx => dx, &
        sw_dy => dy, &
        sw_XCoord => XCoord, &
        sw_YCoord => YCoord, &
        sw_h => h, &
        sw_init, &
        sw_run, &
        sw_final

		
    implicit none

    public SetServices
	
contains

    subroutine SetServices(gcomp, rc)
        type(ESMF_GridComp)  :: gcomp
        integer, intent(out) :: rc
    
        rc = ESMF_SUCCESS
    
        ! NUOPC_Driver registers the generic methods
        call NUOPC_CompDerive(gcomp, model_SetServices, rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out

        call ESMF_GridCompSetEntryPoint(gcomp, ESMF_METHOD_INITIALIZE, &
            userRoutine=InitializeP0, phase=0, rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out
        
		! 用户添加
        call NUOPC_CompSetEntryPoint(gcomp, ESMF_METHOD_INITIALIZE, &
            phaseLabelList=(/"IPDv01p1"/), userRoutine=AdvertiseFields, rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out
			
        ! 用户添加
        call NUOPC_CompSetEntryPoint(gcomp, ESMF_METHOD_INITIALIZE, &
            phaseLabelList=(/"IPDv01p3"/), userRoutine=RealizeFieldsProvidingGrid, rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out

        call NUOPC_CompSpecialize(gcomp, specLabel=model_label_Advance, &
            specRoutine=ModelAdvance, rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out
    
    end subroutine


    subroutine InitializeP0(gcomp, importState, exportState, clock, rc)
        type(ESMF_GridComp)  :: gcomp
        type(ESMF_State)     :: importState, exportState
        type(ESMF_Clock)     :: clock
        integer, intent(out) :: rc
    
        rc = ESMF_SUCCESS
     
        ! Switch to "IPDv01" by filtering all other phaseMap entries
        call NUOPC_CompFilterPhaseMap(gcomp, ESMF_METHOD_INITIALIZE, &
            acceptStringList=(/"IPDv01"/), rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out
    
    end subroutine


    subroutine AdvertiseFields(gcomp, importState, exportState, clock, rc)
        type(ESMF_GridComp)  :: gcomp
        type(ESMF_State)     :: importState, exportState
        type(ESMF_Clock)     :: clock
        integer, intent(out) :: rc
    
        rc = ESMF_SUCCESS

        call NUOPC_FieldDictionaryAddEntry("height", "m", rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out

        call NUOPC_Advertise(exportState, StandardName="height", name="h", rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out

        call sw_init()  ! 初始化
    
    end subroutine


    subroutine RealizeFieldsProvidingGrid(gcomp, importState, exportState, clock, rc)
        type(ESMF_GridComp)  :: gcomp
        type(ESMF_State)     :: importState, exportState
        type(ESMF_Clock)     :: clock
        integer, intent(out) :: rc
    
        type(ESMF_Grid) :: ModelGrid
        type(ESMF_Field) :: height
    
        rc = ESMF_SUCCESS
    
        ModelGrid = CreateGrid_ModelGrid(rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out

        height = ESMF_FieldCreate(name="h", &
            grid=ModelGrid, &
            farray=sw_h, &   ! 指针代入
            indexflag=ESMF_INDEX_GLOBAL, &
            rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out
	
        call NUOPC_Realize(exportState, field=height, rc=rc)   ! 输出height
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out

    end subroutine


    function CreateGrid_ModelGrid(rc)
        type(ESMF_Grid) :: CreateGrid_ModelGrid
        integer, intent(out), optional :: rc

        real(ESMF_KIND_R8), pointer :: coordX(:,:), coordY(:,:)
        integer :: i, j
    
        rc = ESMF_SUCCESS
        CreateGrid_ModelGrid = ESMF_GridCreateNoPeriDim(name="ModelGrid", &
            minIndex=(/1, 1/), &
            maxIndex=(/sw_nx, sw_ny/), &
            indexflag=ESMF_INDEX_GLOBAL, &
            !minCornerCoord=(/1.0_ESMF_KIND_R8, 1.0_ESMF_KIND_R8/), &
            !maxCornerCoord=(/100.0_ESMF_KIND_R8, 100.0_ESMF_KIND_R8/), &
            rc=rc)

        ! add coordinates
        call ESMF_GridAddCoord(CreateGrid_ModelGrid, &
            staggerloc=ESMF_STAGGERLOC_CENTER, rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out

        call ESMF_GridGetCoord(CreateGrid_ModelGrid, coordDim=1, &
            staggerloc=ESMF_STAGGERLOC_CENTER, &
            farrayPtr=coordX, rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out

        call ESMF_GridGetCoord(CreateGrid_ModelGrid, coordDim=2, &
            staggerloc=ESMF_STAGGERLOC_CENTER, &
            farrayPtr=coordY, rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out

        ! set coordinates
        do i=1,sw_nx
            do j=1,sw_ny
                coordX(i,j) = sw_XCoord(i,j)  ! from sw_refactor
                coordY(i,j) = sw_YCoord(i,j)  ! from sw_refactor
            enddo
        enddo
    
    end function    

    function CreateGrid_Uniform(rc)
        type(ESMF_Grid) :: CreateGrid_Uniform
        integer, intent(out), optional :: rc

        rc = ESMF_SUCCESS
        CreateGrid_Uniform = ESMF_GridCreateNoPeriDimUfrm(name="ModelGrid", &
            minIndex=(/1, 1/), &
            maxIndex=(/sw_nx, sw_ny/), &
            !indexflag=ESMF_INDEX_GLOBAL, &
            minCornerCoord=(/0.0_ESMF_KIND_R8, 0.0_ESMF_KIND_R8/), &
            maxCornerCoord=(/(sw_nx-1)*sw_dx, (sw_ny-1)*sw_dy/), &
            staggerloclist=(/ESMF_STAGGERLOC_CENTER, ESMF_STAGGERLOC_CORNER/), &
            coordSys=ESMF_COORDSYS_CART, &
            rc=rc)

    end function

    subroutine ModelAdvance(gcomp, rc)
        type(ESMF_GridComp)  :: gcomp
        integer, intent(out) :: rc

         ! local variables
        type(ESMF_Clock)              :: clock
        type(ESMF_State)              :: importState, exportState
        type(ESMF_Time)               :: currTime
        type(ESMF_TimeInterval)       :: timeStep

        integer(ESMF_KIND_I4)         :: hour, minute
        type(ESMF_Field)              :: height
        integer, save                 :: timeslice = 1
        real(ESMF_KIND_R8),pointer    :: fptr(:,:)

        rc = ESMF_SUCCESS

        ! query the Component for its clock, importState and exportState
        call NUOPC_ModelGet(gcomp, modelClock=clock, importState=importState, &
            exportState=exportState, rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out
    
        ! advance the model: currTime -> currTime + timeStep

        call ESMF_ClockPrint(clock, options="currTime", &
            preString="------>Advancing ShallowWaterCap from: ", rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
           file=__FILE__)) &
            return  ! bail out

       ! call ESMF_ClockPrint(clock, options="stopTime", &
       !     preString="--------------------------------> to: ", rc=rc)
       ! if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
       !     line=__LINE__, &
       !     file=__FILE__)) &
       !     return  ! bail out


        call sw_run()  ! 时间管理后的运行

        call ESMF_ClockGet(clock, currTime=currTime, rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out

        call ESMF_TimeGet(currTime, h=hour, m=minute, rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out

        if (mod(minute,10)==0) then
            ! get height from export state
            call ESMF_StateGet(exportState, "h", height, rc=rc)
            if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
                line=__LINE__, &
                file=__FILE__)) &
                return  ! bail out

            call ESMF_FieldGet(height, farrayPtr=fptr, rc=rc)
            if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
                line=__LINE__, &
                file=__FILE__)) &
                return  ! bail out

            print *, "min/max =", minval(fptr), maxval(fptr)

            call ESMF_FieldWrite(height, fileName="height.nc", &
                variableName="height", overwrite=.true., &
                timeslice=timeslice, rc=rc)
            if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
                line=__LINE__, &
                file=__FILE__)) &
                return  ! bail out

            timeslice = timeslice + 1
        endif


    end subroutine
	
end module