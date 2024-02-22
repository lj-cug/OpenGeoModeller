module sw_driver
	
    use ESMF
    use NUOPC
    use NUOPC_Driver, driver_SetServices => SetServices, &
        driver_label_SetModelServices => label_SetModelServices
		
    use sw_cap, only: sw_SetServices => SetServices

    implicit none

    public SetServices
	
contains

    subroutine SetServices(gcomp, rc)
        type(ESMF_GridComp)  :: gcomp
        integer, intent(out) :: rc
    
        rc = ESMF_SUCCESS
    
        ! NUOPC_Driver registers the generic methods
        call NUOPC_CompDerive(gcomp, driver_SetServices, rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out

        call NUOPC_CompSpecialize(gcomp, specLabel=driver_label_SetModelServices, &
            specRoutine=SetModelServices, rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out
    
    end subroutine


    subroutine SetModelServices(driver, rc)
        type(ESMF_GridComp)  :: driver
        integer, intent(out) :: rc

        type(ESMF_Time)            :: startTime
        type(ESMF_Time)            :: stopTime
        type(ESMF_TimeInterval)    :: timeStep
        type(ESMF_Clock)           :: internalClock

        rc = ESMF_SUCCESS

        call NUOPC_DriverAddComp(driver, compLabel="SW", &
            compSetServicesRoutine=sw_SetServices, rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out

        ! set the model clock
        call ESMF_TimeIntervalSet(timeStep, s=60, rc=rc) ! 1 minute steps
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return

        call ESMF_TimeSet(startTime, yy=2016, mm=9, dd=1, h=0, m=0, rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return

        call ESMF_TimeSet(stopTime, yy=2016, mm=9, dd=5, h=0, m=0, rc=rc)  ! 4 days
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return

        internalClock = ESMF_ClockCreate(name="AppClock", &
            timeStep=timeStep, startTime=startTime, stopTime=stopTime, rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return
  
        call ESMF_GridCompSet(driver, clock=internalClock, rc=rc)
        if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
            line=__LINE__, &
            file=__FILE__)) &
            return  ! bail out


    end subroutine

	
end module


