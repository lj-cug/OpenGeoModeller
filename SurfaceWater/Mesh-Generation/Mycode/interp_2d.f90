program interpolate
!
!  将地形散点插值到三角形网格节点上，形成网格地形
!
implicit none
integer*4,parameter :: points=2282177
integer :: i,j,k
integer*4 :: elem,node
character :: tmp1,tmp2
real*8 :: p   ! IDW power
integer,allocatable,dimension(:,:):: elnode
real*8,allocatable,dimension(:)::x,y,z
real*8,allocatable,dimension(:)::x1,y1,z1

open(unit=1,file='tgr.dat',status='old')      ! mesh data without terrain z value
open(unit=2,file='terrain.dat',status='old')  ! scattered terrain data point
open(unit=3,file='MeshTopo.dat',status='replace')
!
write(*,*)'Begin to read TIN mesh without terrain data...'
read(1,*);read(1,*)
read(1,*)tmp1,node,tmp2,elem
read(1,*)
allocate(x(node),y(node),z(node))
allocate(elnode(elem,3))

do i=1,node
   read(1,*) x(i),y(i),z(i)
enddo
do j=1,elem
   read(1,*) (elnode(j,k),k=1,3)
enddo
!
write(*,*) 'Begin to read the scattered terrain data...'
read(2,*)
allocate(x1(points),y1(points),z1(points))
do i=1,points
   read(2,*)x1(i),y1(i),z1(i)
enddo
!
write(*,*)'Begin to interpolate the scattered data to TIN...'
p = 0.6
! The meshtopo is the interpolated value, scatter data are input value
!call shepard_interp_2d(points,x1,y1,z1,p,node,x,y,z)
call nearest_interp_2d(points,x1,y1,z1,node,x,y,z)
write(*,*)'IDW interpolation end..'

write(*,*)'Write the interpolated data to Tecplot file...'
write(3,*)'VARIABLES = x,y,z'
write(3,*)'ZONE T="triangle"'
write(3,*)'N=',node,',E=',elem,',ZONETYPE=FETRIANGLE'
write(3,*)'DATAPACKING=POINT'
do i=1,node
   write(3,*)x(i),y(i),z(i)
enddo
do j=1,elem
   write(3,*) (elnode(j,k),k=1,3)
enddo

write(*,*)'Scattered data are IDW interpolated to triangular mesh successfully.'

close(1)
close(2)
close(3)
stop
end


subroutine shepard_interp_2d ( nd, xd, yd, zd, p, ni, xi, yi, zi )

!*****************************************************************************80
!
!! SHEPARD_INTERP_2D evaluates a 2D Shepard IDW interpolant.
!
!  Discussion:
!
!    This code should be vectorized.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    21 September 2012
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Donald Shepard,
!    A two-dimensional interpolation function for irregularly spaced data,
!    ACM '68: Proceedings of the 1968 23rd ACM National Conference,
!    ACM, pages 517-524, 1969.
!
!  Parameters:
!
!    Input, integer ( kind = 4 ) ND, the number of data points.
!
!    Input, real ( kind = 8 ) XD(ND), YD(ND), the data points.
!
!    Input, real ( kind = 8 ) ZD(ND), the data values.
!
!    Input, real ( kind = 8 ) P, the power.
!
!    Input, integer ( kind = 4 ) NI, the number of interpolation points.
!
!    Input, real ( kind = 8 ) XI(NI), YI(NI), the interpolation points.
!
!    Output, real ( kind = 8 ) ZI(NI), the interpolated values.
!
  implicit none

  integer ( kind = 4 ) nd
  integer ( kind = 4 ) ni

  integer ( kind = 4 ) i
  integer ( kind = 4 ) j
  real ( kind = 8 ) p
  real ( kind = 8 ) s
  real ( kind = 8 ) w(nd)
  real ( kind = 8 ) xd(nd)
  real ( kind = 8 ) xi(ni)
  real ( kind = 8 ) yd(nd)
  real ( kind = 8 ) yi(ni)
  integer ( kind = 4 ) z
  real ( kind = 8 ) zd(nd)
  real ( kind = 8 ) zi(ni)

  do i = 1, ni

    if ( p == 0.0D+00 ) then
      w(1:nd) = 1.0D+00 / real ( nd, kind = 8 )
    else
    
! Calculate the distance between the node and points
      z = -1
      do j = 1, nd   ! 每个点都要搜索一遍
        w(j) = sqrt ( ( xi(i) - xd(j) ) ** 2 + ( yi(i) - yd(j) ) ** 2 )
        if ( w(j) == 0.0D+00 ) then
          z = j
          exit
        end if              
      end do
! IDW calculation
      if ( z /= -1 ) then
        w(1:nd) = 0.0D+00
        w(z) = 1.0D+00
      else
        w(1:nd) = 1.0D+00 / w(1:nd) ** p
        s = sum ( w )
        w(1:nd) = w(1:nd) / s
      end if
    end if
    zi(i) = dot_product ( w, zd )
    
! Screen show the interpolation process   -LJ
    write(*,*) 'Point #:',i,'Completed=',i/real(ni)*100,'%'
  end do

  return
end subroutine shepard_interp_2d


subroutine nearest_interp_2d ( nd, xd, yd, zd, ni, xi, yi, zi )

  implicit none

  integer ( kind = 4 ) nd
  integer ( kind = 4 ) ni

  integer ( kind = 4 ) i
  integer ( kind = 4 ) j,k
  real ( kind = 8 ) nearest,distance
  real ( kind = 8 ) xd(nd)
  real ( kind = 8 ) xi(ni)
  real ( kind = 8 ) yd(nd)
  real ( kind = 8 ) yi(ni)
  integer ( kind = 4 ) z
  real ( kind = 8 ) zd(nd)
  real ( kind = 8 ) zi(ni)

 do i = 1, ni   
! Calculate the distance between the node and points
      nearest = 1.0e30
      do j = 1, nd   ! 每个点都要搜索一遍
        distance = sqrt ( ( xi(i) - xd(j) ) ** 2 + ( yi(i) - yd(j) ) ** 2 )  
        
        if(distance < nearest)  then
           nearest = distance
           k = j
        endif
                   
      end do
      
 ! The nearest point     
    zi(i) = zd(k)  
    
! Screen show the interpolation process   -LJ
    if(mod(i,100)==0) write(*,*) 'Point #:',i,'Completed=',i/real(ni)*100,'%'
  end do

  return


end subroutine nearest_interp_2d 