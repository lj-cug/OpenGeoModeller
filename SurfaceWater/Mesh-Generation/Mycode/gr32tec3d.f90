module mesh
     implicit none
     public  !Default scope is public 
     
    integer,allocatable,dimension(:) :: kbp  ! 底部垂向分层号
 	real*8,allocatable,dimension(:) :: X,Y,ZB,dp,eta,eta2
	integer,allocatable,dimension(:,:) :: nep
    real*8,allocatable,dimension(:,:) :: znl,scalar  ! 垂向坐标和初始化的标量场
    integer :: nelem, nodes   
    real*8 :: zmsl
     
    !Vertical layer data
    !Vertical coord. types. 2: SZ; -1: Z (only significant in sflux_subs.F90); 1:
    !localized sigma
    integer :: ivcor
    integer :: nvrt                    ! Number of vertical layers
    integer :: kz                      ! Number of Z levels
    integer :: nsig                    ! Number of S levels
    real*8:: h_s,h_c,theta_b,theta_f,s_con1 !constants used in vgrid.in
    real*8,allocatable :: ztot(:)  ! Z coord. of Z levels (local frame)
    real*8,allocatable :: sigma(:) ! sigma coordinates
    real*8,allocatable :: cs(:)    ! function in S-coordinate  
    real*8,allocatable :: dcs(:)   ! derivative of cs()
    real*8 :: h0
end  module mesh
    
program  main
    use mesh
    implicit none
	integer :: i,j,k,l,kin	
	character(len=15) :: filename1,filename2
    
    zmsl=0  ! Sea surface level to test
    h0 = 0.02  
    
 ! Open the input files 
    print *, 'The input gr3 filename:'
  !  read(*,*) filename1
    filename1 = 'hgrid.gr3'
	OPEN(1,FILE=filename1,status='old')  
	
    print *, 'The output tecplot filename:'
  !  read(*,*) filename2
    filename2 = 'hgrid_3D.dat'
	OPEN(2,FILE=filename2,status='replace')

    ! Read the gr3 file
	read(1,*) 
	read(1,*) nelem,nodes
    allocate(kbp(nodes))
	allocate(x(nodes),y(nodes),zb(nodes),dp(nodes),eta(nodes),eta2(nodes))
    allocate(nep(nelem,4))  ! 四边形
    
	do i=1,nodes
	  read(1,*)j,X(I),Y(I),dp(I)
    enddo
	do j=1,nelem
	   read(1,*)k,l,(nep(J,1:3))
       nep(j,4) = nep(j, 3)  ! 三角形->四边形
    enddo
    
    do i=1,nodes
     !  dp(i) = zmsl - ZB(i)   ! 水深
       if(dp(i)<h0) dp(i) = h0
       eta(i) = 0.0  ! 上时刻的水位波动
       eta2(i) = 0.0 ! 当前时刻的水位波动
    enddo
    
  ! Now read the vertical coordinate parameters  
  open(19,file='vgrid.in',status='old')
  
  read(19,*)ivcor
  if(ivcor/=2) then
      write(*,*)'We must use SZ Coor. when creating 3D mesh file.'
      stop
  endif

    read(19,*) nvrt,kz,h_s   ! kz>=1
    if(nvrt<2) then
        write(*,*)'nvrt<2'
        stop
    endif
    if(kz<1) then !.or.kz>nvrt-2) then
      write(*,*)'wrong kz:',kz
      stop
    endif
    if(h_s<6.0) then
      write(*,*)'h_s needs to be larger:',h_s
      stop
    endif

    ! Allocate vertical layers arrays
    allocate(ztot(nvrt),sigma(nvrt),cs(nvrt),dcs(nvrt))

    ! # of z-levels excluding "bottom" at h_s
    read(19,*) !for adding comment "Z levels"
    do k=1,kz-1
      read(19,*)j,ztot(k)
      if(ztot(k)>=-h_s) then
        write(*,*)'Illegal Z level:',k
        stop
      endif
      if(k>1) then; if(ztot(k)<=ztot(k-1)) then
        write(*,*)'z-level inverted:',k
        stop
      endif; endif
    enddo      !k
    read(19,*) !level kz       
    ! In case kz=1, there is only 1 ztot(1)=-h_s
    ztot(kz)=-h_s

    nsig=nvrt-kz+1 ! # of S levels (including "bottom" & f.s.)
    read(19,*)     ! for adding comment "S levels"
    read(19,*)h_c,theta_b,theta_f
    if(h_c<2.0.or.h_c>=h_s) then ! large h_c to avoid 2nd type abnormaty
      write(*,*)'h_c needs to be larger:',h_c
      stop
    endif
    if(theta_b<0.or.theta_b>1) then
      write(*,*)'Wrong theta_b:',theta_b
      stop
    endif
    if(theta_f<=0) then
      write(*,*)'Wrong theta_f:',theta_f
      stop
    endif
    !Pre-compute constants
    s_con1=sinh(theta_f)

    sigma(1)=-1   !bottom
    sigma(nsig)=0 !surface
    read(19,*) !level kz
    do k=kz+1,nvrt-1
      kin=k-kz+1
      read(19,*) j,sigma(kin)
      if(sigma(kin)<=sigma(kin-1).or.sigma(kin)>=0) then
        write(*,*)'Check sigma levels at:',k,sigma(kin),sigma(kin-1)
        stop
      endif
    enddo !k
    read(19,*) !level nvrt
    close(19)

    ! Compute C(s) and C'(s)
    do k=1,nsig
      cs(k)=(1-theta_b)*sinh(theta_f*sigma(k))/sinh(theta_f)+  &
            & theta_b*(tanh(theta_f*(sigma(k)+0.5))-tanh(theta_f*0.5))/2/tanh(theta_f*0.5)
      dcs(k)=(1-theta_b)*theta_f*cosh(theta_f*sigma(k))/sinh(theta_f)+  &
            & theta_b*theta_f/2/tanh(theta_f*0.5)/cosh(theta_f*(sigma(k)+0.5))**2
    enddo !k
  
! 计算垂向坐标
    allocate(znl(nvrt, nodes),scalar(nvrt, nodes))
    do i=1, nodes
      !  do k=1, nvrt
      !     znl(k,i) =  dp(i)/(nvrt-1)*(k-1)
      !  enddo
        call zcoor(i, kbp(i), znl(:,i))  ! 计算节点上的垂向坐标    
    enddo

  ! Write 3D Tecplot file  
     write(2,*)'TITLE="3D_Mesh"'
     write(2,*) 'VARIABLES="x","y","z","conc"'
     write(2,*) 'ZONE T="VOLUME",','NODES=', nodes*nvrt,',ELEMENTS=',nelem*(nvrt-1),',  &
    &  DATAPACKING=POINT,ZONETYPE=FEBRICK'
              
    do k=1,nvrt	
	   do i=1,nodes
         write(2,10) x(i),y(i),znl(k,i),0.0
	   enddo
    enddo
    
  	do k=1,nvrt-1
       do i=1,nelem			   		
		write(2,11)(nep(i,j)+(k-1)*nodes,j=1,4),(nep(i,j)+k*nodes,j=1,4)  ! 1个brick有8个节点
	   enddo
	enddo  
10 format(2(f15.6,1x),2(f12.3),1x)
11 format(8(I7,1x))
   close(2)
   
   deallocate(x,y,ZB,dp,eta,eta2)
   deallocate(znl, scalar)
	write(*,*)"data change complete normally!"
	stop
    end

    !!!!!!!
subroutine zcoor(inode,kbpl,ztmp)
  use mesh
  implicit none
  integer, intent(in) :: inode 
  integer, intent(out) :: kbpl
  real*8, intent(out) :: ztmp(nvrt)
  
!     Local
   integer :: k,kin,m
   real*8 :: hmod2,z0,z_1,sp,tmp
  
   ! Now, we create the real vertical coordinate
  ! WARNING: explicitly specify bottom/surface to avoid underflow
        hmod2=min(dp(inode),h_s)
        ztmp(kz)=-hmod2   ! to avoid underflow
        ztmp(nvrt)=eta2(inode)

        do k=kz+1,nvrt-1
          kin=k-kz+1
          if(hmod2<=h_c) then
            ztmp(k)=sigma(kin)*(hmod2+eta2(inode))+eta2(inode)
           !todo: assert
          else if(eta2(inode)<=-h_c-(dp(inode)-h_c)*theta_f/s_con1) then
            write(*,*)'ZCOOR: Pls choose a larger h_c:',eta(inode),h_c
            stop
          else
            ztmp(k)=eta2(inode)*(1+sigma(kin))+h_c*sigma(kin)+(hmod2-h_c)*cs(kin)
          endif
        enddo !k

        if(dp(inode)<=h_s) then
          kbpl=kz
        else !z levels
!         Find bottom index
          kbpl=0
          do k=1,kz-1
            if(-dp(inode)>=ztot(k).and.-dp(inode)<ztot(k+1)) then
              kbpl=k
              exit
            endif
          enddo !k
          !todo: assert
          if(kbpl==0) then
            write(*,*)'ZCOOR: Cannot find a bottom level:',dp(inode)
            stop
          endif
          ztmp(kbpl)=-dp(inode)
          do k=kbpl+1,kz-1
            ztmp(k)=ztot(k)
          enddo !k
        endif !dep<=h_s
  
      do k=kbpl+1,nvrt
        !todo: assert
        if(ztmp(k)-ztmp(k-1)<=0) then
            write(*,*)'ZCOOR: Inverted z-level:',ivcor,k,kbpl,eta2(inode),dp(inode),ztmp(k),ztmp(k-1)
            stop
        endif
      enddo !k
end subroutine