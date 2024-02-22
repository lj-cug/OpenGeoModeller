module ATM
	
    use ESMF
    use NUOPC
    use NUOPC_Model, model_SetServices => SetServices, &
        model_label_Advance => label_Advance
		
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

        call NUOPC_CompSetEntryPoint(gcomp, ESMF_METHOD_INITIALIZE, &
            phaseLabelList=(/"IPDv01p1"/), userRoutine=AdvertiseFields, rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out

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

        call NUOPC_Advertise(importState, StandardName="height", name="h", rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out
    
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

        ! field "ImportField"
        height = ESMF_FieldCreate(name="h", grid=ModelGrid, &
            typekind=ESMF_TYPEKIND_R8, rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out
	
        call NUOPC_Realize(importState, field=height, rc=rc)  ! ÊäÈëheight
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out
    
    end subroutine


    function CreateGrid_ModelGrid(rc)
        type(ESMF_Grid) :: CreateGrid_ModelGrid
        integer, intent(out), optional :: rc
    
        rc = ESMF_SUCCESS
        CreateGrid_ModelGrid = ESMF_GridCreateNoPeriDimUfrm(name="ModelGrid", &
            minIndex=(/1, 1/), &
            maxIndex=(/254*2, 50*2/), &
            minCornerCoord=(/0.0_ESMF_KIND_R8, 0.0_ESMF_KIND_R8/), &
            maxCornerCoord=(/253*100.0e3_ESMF_KIND_R8, 49*100.0e3_ESMF_KIND_R8/), &
            staggerloclist=(/ESMF_STAGGERLOC_CENTER, ESMF_STAGGERLOC_CORNER/), &
            coordSys=ESMF_COORDSYS_CART, &
            rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out
    
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
            preString="------>Advancing ATM from: ", rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out

        call ESMF_ClockPrint(clock, options="stopTime", &
            preString="--------------------------------> to: ", rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out


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

        if (mod(minute,30)==0) then
            ! get height from export state
            call ESMF_StateGet(importState, "h", height, rc=rc)
            if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
                line=__LINE__, &
                file=__FILE__)) &
                return  ! bail out

            call ESMF_FieldGet(height, farrayPtr=fptr, rc=rc)
            if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
                line=__LINE__, &
                file=__FILE__)) &
                return  ! bail out

            !print *, "min/max =", minval(fptr), maxval(fptr)

            call ESMF_FieldWrite(height, fileName="atm_height.nc", &
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


