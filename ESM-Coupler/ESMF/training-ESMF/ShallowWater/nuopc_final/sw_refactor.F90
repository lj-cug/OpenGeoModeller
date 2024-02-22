! Shallow Water Model
!
! This code is based on a Matlab code by Robin Hogan:
! http://www.met.reading.ac.uk/~swrhgnrj/shallow_water_model/
!
! The Matlab has essentially been transcribed into Fortran, retaining
! the same numerics, parameters, and code structure.  Currently, the
! only parameters supported are:
!
!   orography = FLAT
!   initial_conditions = GAUSSIAN_BLOB
!   initially_geostrophic = .false.
!   add_random_height_noise = .false.
!
! Unlike the  original code, NetCDF is used to save output.
!
! Author: Rocky Dunlap  <rocky.dunlap@noaa.gov>

module sw_refactor

#ifdef WITH_NETCDF
    use netcdf
#endif
    implicit none

    private

    ! interface for function to find mean of values in an array
    !interface
    !    function mean(arr)
    !        real(8), intent(inout), dimension(:) :: arr
    !        real(8) :: mean
    !    end function
    !end interface


    !!!!!!!!!!!!!!!!!!!!!!!!
    ! Section 0: Constants !
    !!!!!!!!!!!!!!!!!!!!!!!!

    ! Possible initial conditions of the height field
    integer, parameter :: UNIFORM_WESTERLY = 1
    integer, parameter :: ZONAL_JET = 2
    integer, parameter :: REANALYSIS = 3
    integer, parameter :: GAUSSIAN_BLOB = 4
    integer, parameter :: STEP = 5
    integer, parameter :: CYCLONE_IN_WESTERLY = 6
    integer, parameter :: SHARP_SHEAR = 7
    integer, parameter :: EQUATORIAL_EASTERLY = 8
    integer, parameter :: SINUSOIDAL = 9

    ! Possible orographies
    integer, parameter :: FLAT = 0
    integer, parameter :: SLOPE = 1
    integer, parameter :: GAUSSIAN_MOUNTAIN = 2
    integer, parameter :: EARTH_OROGRAPHY = 3
    integer, parameter :: SEA_MOUNT = 4

    !!!!!!!!!!!!!!!!!!!!!!!!!
    ! Section 1: Parameters !
    !!!!!!!!!!!!!!!!!!!!!!!!!

    real(8), parameter :: g = 9.81         ! Acceleration due to gravity (m/s2)
    real(8), parameter :: f = 1.0e-4       ! Coriolis parameter (s-1)
    real(8), parameter :: beta = 1.6e-11   ! Meridional gradient of f (s-1m-1)

    real(8), parameter :: dt_mins = 1                  ! Timestep (minutes)
    real(8), parameter :: output_interval_mins = 5    ! Time between outputs (minutes)
    real(8), parameter :: forecast_length_days = 4     ! Total simulation length (days)
    character(*), parameter :: output_file = "swout.nc"  ! netcdf output file

    real(8), parameter :: orography = FLAT
    real(8), parameter :: initial_conditions = GAUSSIAN_BLOB
    logical, parameter :: initially_geostrophic = .false.
    logical, parameter :: add_random_height_noise = .false.

    ! If you change the number of gridpoints then orography=EARTH_OROGRAPHY
    ! or initial_conditions=REANALYSIS won't work
    integer, parameter :: nx = 254      ! Number of zonal gridpoints
    integer, parameter :: ny = 50       ! Number of meridional gridpoints

    real(8), parameter :: dx=100.0e3    ! Zonal grid spacing (m)
    real(8), parameter :: dy=dx         ! Meridional grid spacing


    !!!!!!!!!!!!!!!!!!!!!!!!!
    ! Section 2: Setup      !
    !!!!!!!!!!!!!!!!!!!!!!!!!

    integer :: i
    !integer :: rc   ! return code
    integer :: n    ! current timestep

    ! netcdf handles
    integer :: ncid ! id of netcdf file
    integer :: x_dimid, y_dimid, t_dimid
    integer :: var_dimids(3), coord_dimids(2)
    integer :: u_varid, v_varid, h_varid, x_varid, y_varid

    real(8), parameter :: dt = dt_mins * 60.0       ! Timestep (s)
    real(8), parameter :: output_interval = output_interval_mins * 60.0        ! Time between outputs (s)
    real(8), parameter :: forecast_length = forecast_length_days * 24.0 * 3600.0   ! Forecast length (s)

    integer, parameter :: nt = floor(forecast_length/dt) + 1     ! Number of timesteps
    integer, parameter :: timesteps_between_outputs = floor(output_interval/dt)
    integer, parameter :: noutput = ceiling(1.0 * nt / timesteps_between_outputs) ! Number of output frames

    real(8), dimension(nx) :: x             ! Zonal distance coordinate (m)
    real(8), dimension(ny) :: y             ! Meridional distance coordinate (m)
    real(8), dimension(nx, ny) :: XCoord    ! Matrix of coordinate variables
    real(8), dimension(nx, ny) :: YCoord    ! Matrix of coordinate variables
    real(8), dimension(nx, ny) :: ORO       ! Orography field
    real(8), dimension(nx, ny) :: height    ! height of fluid
    real(8), dimension(nx, ny) :: FCor      ! Coriolis parameter as a matrix of values varying in y only
    real(8), dimension(nx, ny) :: u         ! wind velocity (m/s)
    real(8), dimension(nx, ny) :: v         ! wind velocity (m/s)
    real(8), dimension(nx, ny) :: h         ! depth of fluid (height - ORO)
    real(8), dimension(2:nx-1, 2:ny-1) :: u_accel   ! wind acceleration (m/s^2)
    real(8), dimension(2:nx-1, 2:ny-1) :: v_accel   ! wind acceleration (m/s^2)

    real(8), dimension(nx * ny) :: u_temp
    real(8), dimension(nx * ny) :: v_temp

    real(8), dimension(2:nx-1, 2:ny-1) :: unew
    real(8), dimension(2:nx-1, 2:ny-1) :: vnew
    real(8), dimension(2:nx-1, 2:ny-1) :: hnew

    real(8), dimension(nx, ny, noutput) :: u_save    ! save u data
    real(8), dimension(nx, ny, noutput) :: v_save    ! save v data
    real(8), dimension(nx, ny, noutput) :: h_save    ! save h data
    real(8), dimension(noutput) :: t_save    ! save t data



    real(8) :: std_blob     ! Standard deviation of blob (m), orography = GAUSSIAN_BLOB
    real(8) :: max_u        ! max wind velocity
    integer :: i_save       ! index of saved field data

    ! make public what needs to be accessed by the cap
    public :: nx        ! used by cap
    public :: ny        ! used by cap
    public :: dx
    public :: dy
    public :: XCoord    ! used by cap
    public :: YCoord    ! used by cap
    public :: h         ! fluid depth, will be exported by cap
    public :: sw_init   ! called by cap
    public :: sw_run    ! called by cap
    public :: sw_final  ! called by cap

    contains

    subroutine sw_init()
        x = (/ (i, i=0, nx-1) /) * dx
        y = (/ (i, i=0, ny-1) /) * dy

        do i=1,ny
            XCoord(:,i) = x
        end do

        do i=1,nx
            YCoord(i,:) = y
        end do

        !DEBUG
        !print *, "nt=", nt
        !print *, "timesteps_between_outputs=", timesteps_between_outputs
        !print *, "noutput=", noutput
        !print *, "x=",x
        !print *, "y=", y
        !print *, "XCoord=", XCoord(:,1)
        !print *, "YCoord=", YCoord(1,:)
        !print *, "sum(x) = ", sum(x)
        !print *, "size(x) = ", size(x)
        !print *, "mean(x) = ", mean(x)
        !print *, "mean(y) = ", mean(y)

        if (orography == FLAT) then
            ORO = 0.0
        else
            print *, "That orography parameter setting is currently not supported."
            stop
        end if


    !% Create the orography field "H"
    !switch orography
    !  case FLAT
    !    H = zeros(nx, ny);
    !  case SLOPE
    !    H = 9000.*2.*abs((mean(x)-X)./max(x));
    !  case GAUSSIAN_MOUNTAIN
    !    std_mountain_x = 5.*dx; % Std. dev. of mountain in x direction (m)
    !    std_mountain_y = 5.*dy; % Std. dev. of mountain in y direction (m)
    !    H = 4000.*exp(-0.5.*((X-mean(x))./std_mountain_x).^2 ...
    !                  -0.5.*((Y-mean(y))./std_mountain_y).^2);
    !  case SEA_MOUNT
    !    std_mountain = 40.0.*dy; % Standard deviation of mountain (m)
    !    H = 9250.*exp(-((X-mean(x)).^2+(Y-0.5.*mean(y)).^2)./(2*std_mountain^2));
    !  case EARTH_OROGRAPHY
    !    load digital_elevation_map.mat
    !    H = elevation;
    !    % Enforce periodic boundary conditions in x
    !    H([1 end],:)=H([end-1 2],:);
    ! otherwise
    !   error(['Don''t know what to do with orography=' num2str(orography)]);
    !end
    !

        if (initial_conditions == GAUSSIAN_BLOB) then
            std_blob = 8.0 * dy     ! Standard deviation of blob (m)
            height = 9750 + 1000 * exp(-((XCoord-0.25*mean(x))**2 + (YCoord - mean(y))**2) / (2 * std_blob**2))
            !print *, "height=", height(:,1)
        else
            print *, "That initial_conditions parameter setting is currently not supported."
            stop
        end if

    !% Create the initial height field
    !switch initial_conditions
    !  case UNIFORM_WESTERLY
    !    mean_wind_speed = 20; % m/s
    !    height = 10000-(mean_wind_speed*f/g).*(Y-mean(y));
    !  case SINUSOIDAL
    !    height = 10000-350.*cos(Y./max(y).*4.*pi);
    !  case EQUATORIAL_EASTERLY
    !    height = 10000 - 50.*cos((Y-mean(y)).*4.*pi./max(y));
    !  case ZONAL_JET
    !    height = 10000 - tanh(20.0.*((Y-mean(y))./max(y))).*400;
    !  case REANALYSIS
    !    load reanalysis.mat
    !    height = 0.99.*pressure./g;
    ! case GAUSSIAN_BLOB
    !   std_blob = 8.0.*dy; % Standard deviation of blob (m)
    !   height = 9750 + 1000.*exp(-((X-0.25.*mean(x)).^2+(Y-mean(y)).^2)./(2* ...
    !                                                     std_blob^2));
    ! case STEP
    !  height = 9750.*ones(nx, ny);
    !  height(find(X<max(x)./5 & Y>max(y)/10 & Y<max(y).*0.9)) = 10500;
    ! case CYCLONE_IN_WESTERLY
    !   mean_wind_speed = 20; % m/s
    !   std_blob = 7.0.*dy; % Standard deviation of blob (m)
    !    height = 10000-(mean_wind_speed*f/g).*(Y-mean(y)) ...
    !       - 500.*exp(-((X-0.5.*mean(x)).^2+(Y-mean(y)).^2)./(2*std_blob^2));
    !    max_wind_speed = 20; % m/s
    !    height = 10250-(max_wind_speed*f/g).*(Y-mean(y)).^2./max(y) ...
    !       - 1000.*exp(-(0.25.*(X-1.5.*mean(x)).^2+(Y-0.5.*mean(y)).^2)./(2*std_blob^2));
    !  case SHARP_SHEAR
    !    mean_wind_speed = 50; % m/s
    !    height = (mean_wind_speed*f/g).*abs(Y-mean(y));
    !    height = 10000+height-mean(height(:));
    !otherwise
    !   error(['Don''t know what to do with initial_conditions=' num2str(initial_conditions)]);
    !end
    !
    !

        ! Coriolis parameter as a matrix of values varying in y only
        FCor = f + (beta * (YCoord - mean(y)))
        !print *, "FCor=", FCor(:,10)

        ! Initialize the wind to rest
        u = 0.0
        v = 0.0

        !print *, "u=", u
        !print *, "maxval(u)=", maxval(reshape(u, (/ nx*ny /)))

        ! We may need to add small-amplitude random noise in order to initialize
        ! instability
        if (add_random_height_noise) then
            !  height = height + 1.0.*randn(size(height)).*(dx./1.0e5).*(abs(F)./1e-4);
            print *, "The add_random_height_noise parameter is not currently supported."
        end if
    !
    !
    !if initially_geostrophic
    !   % Centred spatial differences to compute geostrophic wind
    !   u(:,2:end-1) = -(0.5.*g./(F(:,2:end-1).*dx)) ...
    !       .* (height(:,3:end)-height(:,1:end-2));
    !   v(2:end-1,:) = (0.5.*g./(F(2:end-1,:).*dx)) ...
    !       .* (height(3:end,:)-height(1:end-2,:));
    !   % Zonal wind is periodic so set u(1) and u(end) as dummy points that
    !   % replicate u(end-1) and u(2), respectively
    !   u([1 end],:) = u([2 end-1],:);
    !   % Meridional wind must be zero at the north and south edges of the
    !   % channel
    !   v(:,[1 end]) = 0;
    !
    !   % Don't allow the initial wind speed to exceed 200 m/s anywhere
    !   max_wind = 200;
    !   u(find(u>max_wind)) = max_wind;
    !   u(find(u<-max_wind)) = -max_wind;
    !   v(find(v>max_wind)) = max_wind;
    !   v(find(v<-max_wind)) = -max_wind;
    !end

        if (initially_geostrophic) then
            print *, "The initially_geostrophic parameter is not currently supported."
        end if

        ! Define h as the depth of the fluid (whereas "height" is the height of
        ! the upper surface)
        h = height - ORO
        !print *, "h=", h(1,:)

        ! Initialize the 3D arrays where the output data will be stored
        u_save = 0.0
        v_save = 0.0
        h_save = 0.0
        t_save = 0.0

        ! Index to stored data
        i_save = 1


        !!!!!!!!!!!!!!!!!!!!!!!!!
        ! Section 2: Main loop  !
        !!!!!!!!!!!!!!!!!!!!!!!!!

#ifdef WITH_NETCDF
        ! set up netcdf
        call check( nf90_create(output_file, NF90_CLOBBER, ncid) )
        call check( nf90_def_dim(ncid, "x", nx, x_dimid) )
        call check( nf90_def_dim(ncid, "y", ny, y_dimid) )
        call check( nf90_def_dim(ncid, "t", noutput, t_dimid) )

        var_dimids = (/ x_dimid, y_dimid, t_dimid /)
        call check( nf90_def_var(ncid, "u", NF90_DOUBLE, var_dimids, u_varid) )
        call check( nf90_def_var(ncid, "v", NF90_DOUBLE, var_dimids, v_varid) )
        call check( nf90_def_var(ncid, "h", NF90_DOUBLE, var_dimids, h_varid) )

        coord_dimids = (/ x_dimid, y_dimid /)
        call check( nf90_def_var(ncid, "x", NF90_DOUBLE, coord_dimids, x_varid) )
        call check( nf90_def_var(ncid, "y", NF90_DOUBLE, coord_dimids, y_varid) )

        call check( nf90_enddef(ncid) )
#endif
    end subroutine sw_init

    ! integration loop will be in driver

    !do n=1, nt
        !  Every fixed number of timesteps we store the fields

        !print *, "n=", n
    subroutine sw_run()

        if (mod(n-1, timesteps_between_outputs) == 0) then
            u_temp = reshape(u, (/nx*ny/))
            v_temp = reshape(v, (/nx*ny/))
            max_u = sqrt(maxval(u_temp*u_temp + v_temp*v_temp))

            print *, "Time = ", (n-1)*dt/3600, " hours (max ", forecast_length_days*24, ")"
            u_save(:,:,i_save) = u
            v_save(:,:,i_save) = v
            h_save(:,:,i_save) = h
            t_save(i_save) = (n-1) * dt
            i_save = i_save + 1

            !print *, "height min/max = ", minval(h), maxval(h)

        end if

        !  Compute the accelerations
        u_accel = FCor(2:nx-1,2:ny-1) * v(2:nx-1,2:ny-1) - (g/(2.0*dx)) * (ORO(3:nx,2:ny-1) - ORO(1:nx-2,2:ny-1))
        v_accel = -FCor(2:nx-1,2:ny-1) * u(2:nx-1,2:ny-1) - (g/(2.0*dy)) * (ORO(2:nx-1,3:ny) - ORO(2:nx-1,1:ny-2))

        !  Call the Lax-Wendroff scheme to move forward one timestep
         call lax_wendroff(nx, ny, dx, dy, dt, g, u, v, h, u_accel, v_accel, unew, vnew, hnew)

        !  Update the wind and height fields, taking care to enforce
        !  boundary conditions
        !  u = unew([end 1:end 1],[1 1:end end]);
        !  v = vnew([end 1:end 1],[1 1:end end]);
        !  v(:,[1 end]) = 0;
        !  h(:,2:end-1) = h_new([end 1:end 1],:);


        u(2:nx-1, 2:ny-1) = unew

        u(1,1) = unew(nx-1,2)
        u(2:nx-1,1) = unew(:,2)
        u(nx,1) = unew(2,2)
        u(1,ny) = unew(nx-1,ny-1)
        u(2:nx-1,ny) = unew(:,ny-1)
        u(nx,ny) = unew(2,ny-1)
        u(1,2:ny-1) = unew(nx-1,:)
        u(nx,2:ny-1) = unew(2,:)


        v(2:nx-1, 2:ny-1) = vnew

        v(1,1) = vnew(nx-1,2)
        v(2:nx-1,1) = vnew(:,2)
        v(nx,1) = vnew(2,2)
        v(1,ny) = vnew(nx-1,ny-1)
        v(2:nx-1,ny) = vnew(:,ny-1)
        v(nx,ny) = vnew(2,ny-1)
        v(1,2:ny-1) = vnew(nx-1,:)
        v(nx,2:ny-1) = vnew(2,:)

        v(:,1) = 0
        v(:,ny) = 0

        h(1,      2:ny-1) = hnew(nx-1,:)
        h(2:nx-1, 2:ny-1) = hnew
        h(nx,     2:ny-1) = hnew(2,:)

    end subroutine sw_run

    !end do

    subroutine sw_final
#ifdef WITH_NETCDF
        call check( nf90_put_var(ncid, x_varid, XCoord) )
        call check( nf90_put_var(ncid, y_varid, YCoord) )

        call check( nf90_put_var(ncid, u_varid, u_save) )
        call check( nf90_put_var(ncid, v_varid, v_save) )
        call check( nf90_put_var(ncid, h_varid, h_save) )

        call check( nf90_close(ncid) )
#endif
    end subroutine sw_final

    ! DEBUG
    !u_temp = reshape(u, (/nx*ny/))
    !v_temp = reshape(v, (/nx*ny/))
    !max_u = sqrt(maxval(u_temp*u_temp + v_temp*v_temp))

    !print *, "maxval(u_temp) = ", maxval(u_temp)
    !print *, "maxval(v_temp) = ", maxval(v_temp)
    !print *, "max_u = ", max_u


    function mean(arr)
        real(8), intent(inout), dimension(:) :: arr
        real(8) :: mean
        mean = sum(arr) / size(arr)
    end function


    subroutine lax_wendroff(nx, ny, dx, dy, dt, g, u, v, h, u_tendency, v_tendency, u_new, v_new, h_new)

        integer, intent(in) :: nx, ny
        real(8), intent(in) :: dx, dy, dt, g
        real(8), dimension(nx, ny), intent(in) :: u, v, h
        real(8), dimension(2:nx-1, 2:ny-1), intent(in) :: u_tendency, v_tendency
        real(8), dimension(2:nx-1, 2:ny-1), intent(out) :: u_new, v_new, h_new

        real(8), dimension(nx, ny) :: uh, vh
        real(8), dimension(nx, ny) :: h_mid_xt, h_mid_yt
        real(8), dimension(nx, ny) :: Ux, Uy
        real(8), dimension(nx, ny) :: uh_mid_xt, uh_mid_yt
        real(8), dimension(nx, ny) :: Vx, Vy
        real(8), dimension(nx, ny) :: vh_mid_xt, vh_mid_yt
        real(8), dimension(nx, ny) :: Ux_mid_xt, Uy_mid_yt
        real(8), dimension(2:nx-1, 2:ny-1) :: uh_new
        real(8), dimension(nx, ny) :: Vx_mid_xt, Vy_mid_yt
        real(8), dimension(2:nx-1, 2:ny-1) :: vh_new

        !print *, "Entering lax_wendroff"


        ! This function performs one timestep of the Lax-Wendroff scheme
        ! applied to the shallow water equations
        !
        ! First work out mid-point values in time and space

        uh = u * h
        vh = v * h

        h_mid_xt = 0.0
        h_mid_yt = 0.0
        uh_mid_xt = 0.0
        uh_mid_yt = 0.0
        vh_mid_xt = 0.0
        vh_mid_yt = 0.0
        
        h_mid_xt(1:nx-1,:) = 0.5 * (h(2:nx,:)+h(1:nx-1,:)) -(0.5*dt/dx) * (uh(2:nx,:)-uh(1:nx-1,:))
        h_mid_yt(:,1:ny-1) = 0.5 * (h(:,2:ny)+h(:,1:ny-1)) -(0.5*dt/dy) * (vh(:,2:ny)-vh(:,1:ny-1))

        Ux = uh * u + 0.5 * g * h**2
        Uy = uh * v
        uh_mid_xt(1:nx-1,:) = 0.5 * (uh(2:nx,:)+uh(1:nx-1,:)) -(0.5*dt/dx) * (Ux(2:nx,:)-Ux(1:nx-1,:))
        uh_mid_yt(:,1:ny-1) = 0.5 * (uh(:,2:ny)+uh(:,1:ny-1)) -(0.5*dt/dy) * (Uy(:,2:ny)-Uy(:,1:ny-1))

        Vx = Uy
        Vy = vh * v + 0.5 * g * h**2
        vh_mid_xt(1:nx-1,:) = 0.5 * (vh(2:nx,:) + vh(1:nx-1,:)) -(0.5*dt/dx) * (Vx(2:nx,:) - Vx(1:nx-1,:))
        vh_mid_yt(:,1:ny-1) = 0.5 * (vh(:,2:ny) + vh(:,1:ny-1)) -(0.5*dt/dy) * (Vy(:,2:ny) - Vy(:,1:ny-1))


        ! Now use the mid-point values to predict the values at the next
        ! timestep

        h_new = 0.0
        u_new = 0.0
        v_new = 0.0
        
        h_new = h(2:nx-1,2:ny-1) &
              - (dt/dx) * (uh_mid_xt(2:nx-1,2:ny-1) - uh_mid_xt(1:nx-2,2:ny-1)) &
              - (dt/dy) * (vh_mid_yt(2:nx-1,2:ny-1) - vh_mid_yt(2:nx-1,1:ny-2))


        Ux_mid_xt = uh_mid_xt * uh_mid_xt / h_mid_xt + 0.5 * g * h_mid_xt**2
        Uy_mid_yt = uh_mid_yt * vh_mid_yt / h_mid_yt


        uh_new = uh(2:nx-1,2:ny-1) &
          - (dt/dx) * (Ux_mid_xt(2:nx-1,2:ny-1) - Ux_mid_xt(1:nx-2,2:ny-1)) &
          - (dt/dy) * (Uy_mid_yt(2:nx-1,2:ny-1) - Uy_mid_yt(2:nx-1,1:ny-2)) &
          + dt * u_tendency * 0.5 * (h(2:nx-1,2:ny-1) + h_new)


        Vx_mid_xt = uh_mid_xt * vh_mid_xt / h_mid_xt
        Vy_mid_yt = vh_mid_yt * vh_mid_yt / h_mid_yt + 0.5 * g * h_mid_yt**2

        vh_new = vh(2:nx-1,2:ny-1) &
          - (dt/dx) * (Vx_mid_xt(2:nx-1,2:ny-1) - Vx_mid_xt(1:nx-2,2:ny-1)) &
          - (dt/dy) * (Vy_mid_yt(2:nx-1,2:ny-1) - Vy_mid_yt(2:nx-1,1:ny-2)) &
          + dt * v_tendency * 0.5 * (h(2:nx-1,2:ny-1) + h_new)

        u_new = uh_new / h_new
        v_new = vh_new / h_new


        ! DEBUG
        !print *, "sum(u) = ", sum(reshape(u, (/nx*ny/)))
        !print *, "sum(u_tendency) = ", sum(reshape(u_tendency, (/(nx-2)*(ny-2)/)))
        !print *, "sum(uh) = ", sum(reshape(uh, (/nx*ny/)))
        !print *, "sum(uh_mid_xt) = ", sum(reshape(uh_mid_xt, (/nx*ny/)))
        !print *, "sum(uh_mid_yt) = ", sum(reshape(uh_mid_yt, (/nx*ny/)))
        !print *, "sum(vh_mid_xt) = ", sum(reshape(vh_mid_xt, (/nx*ny/)))
        !print *, "sum(vh_mid_yt) = ", sum(reshape(vh_mid_yt, (/nx*ny/)))
        !print *, "maxval(Ux_mid_xt) = ", maxval(reshape(Ux_mid_xt, (/nx*ny/)))
        !print *, "maxval(Uy_mid_yt) = ", maxval(reshape(Uy_mid_yt, (/nx*ny/)))
        !print *, "max(u_new) = ", maxval(reshape(u_new, (/(nx-2)*(ny-2)/)))
        !print *, "max(v_new) = ", maxval(reshape(v_new, (/(nx-2)*(ny-2)/)))

        !print *, "Leaving lax_wendroff"

    end subroutine

#ifdef WITH_NETCDF
    ! NetCDF status check
    subroutine check(status)
        use netcdf
        integer, intent ( in) :: status

        if(status /= nf90_noerr) then
          print *, trim(nf90_strerror(status))
          stop "Stopped due to NetCDF error"
        end if
    end subroutine check
#endif



end module sw_refactor



