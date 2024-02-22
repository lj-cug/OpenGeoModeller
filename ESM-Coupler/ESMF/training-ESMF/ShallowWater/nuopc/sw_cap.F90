module sw_cap
	
    use ESMF
    use NUOPC
    use NUOPC_Model,    &
	    model_SetServices => SetServices, &
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
			
        ! Èë¿Úµã
        call ESMF_GridCompSetEntryPoint(gcomp, ESMF_METHOD_INITIALIZE, &
            userRoutine=InitializeP0, phase=0, rc=rc)
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

	
end module


