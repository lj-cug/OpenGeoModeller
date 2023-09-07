  program main
	implicit none
	character filename*50
	integer :: temp1,temp2,temp3
	integer :: numnodes,numeles,numelep1,numelep2
	integer :: NGRPS,NBSETS,NDFCD,NDFVL
	integer i,j,k,l,m,n,o,p,q
    real(kind=8) :: xtemp,ytemp
	real(kind=8) :: temp
	real(kind=8),parameter :: pai=3.1415926
!-------------------------------------------------------------------
!changjiang - inlet
	character(10) :: inlet_name1
    integer :: inlet_node_numbers1  
	integer,dimension(50) :: inlet_node1
!jialingjiang - inlet
	character(10) :: inlet_name2
    integer :: inlet_node_numbers2  
	integer,dimension(50) :: inlet_node2
!wujiang - inlet
	character(10) :: inlet_name3
    integer :: inlet_node_numbers3  
	integer,dimension(50) :: inlet_node3
!Three gorges dam -- outlet
	character(10) :: outlet_name
    integer :: outlet_node_numbers  
	integer,dimension(200) :: outlet_node
!-------------------------------------------------------------------
! left_bank from cj to jialiang
	character(10) :: left_bank_name1
    integer :: left_bank_node_numbers1 
	integer,dimension(100000) :: left_bank_node1
	integer :: pp
	integer,dimension(1000) :: error_p
! left_bank from jialingjiang to tgr dam
	character(10) :: left_bank_name2
    integer :: left_bank_node_numbers2 
	integer,dimension(100000) :: left_bank_node2
!---------------------------------------------------------------------
! right_bank from cj to wujiang1
	character(10) :: right_bank_name1
    integer :: right_bank_node_numbers1  
	integer,dimension(100000) :: right_bank_node1
! right_bank from cj to wujiang2
	character(10) :: right_bank_name2
    integer :: right_bank_node_numbers2  
	integer,dimension(100000) :: right_bank_node2
! right_bank from wujiang to Dam
	character(10) :: right_bank_name3
    integer :: right_bank_node_numbers3  
	integer,dimension(100000) :: right_bank_node3
!----------------------------------------------------------------------
! island -    1 ~ 9 
	character(10),dimension(9):: island_name
    integer,dimension(9) :: island_node_numbers  
	integer,dimension(9,2000) :: island_node

! matrix
	real(kind=8),allocatable:: x(:),y(:),z(:)
	integer,allocatable::nNodes(:,:),nEle(:),nNode(:)
!-----------------------------------------
    filename = 'tgr_depth_80_30m_175.dat'  ! Tecplot meshing topography
	open(unit=1,file=filename,status='old')

!	filename='hlm_2m.neu'   !Gambit导出的网格文件  Polyflow求解器格式
	filename='mesh_80_30m.neu'
	open(unit=2,file=filename,status='old')

    filename='hgrid.gr3'  ! gr3 mesh file for SELFE or SCHISM
	open(unit=3,file=filename)
    write(3,*) 'hgrid.gr3'

    filename='Boundnodes.dat'  ! gr3 mesh file for SELFE or SCHISM
	open(unit=4,file=filename)

!  read the neu file
    do m=1,9 ! include the first 9 lines in neu file
	  if (m.ne.7) then
          read(2,*) !dataTemp
	  else
          read(2,*) numNodes, numEles,NGRPS,NBSETS,NDFCD,NDFVL
	    allocate(x(numNodes),y(numNodes),z(numNodes),nNode(numNodes),nNodes(3,numEles),nEle(numEles))   	     
	  endif
	enddo

	write(3,*) numEles,numNodes
 
 ! read header of tecplot file   
	do m=1,9
	  read(1,*)
	enddo 

	do i=1, numNodes
      read (1,*) xtemp, ytemp, z(i)      ! 只要地形，不要xy坐标
	enddo
  close(1)
write(*,*) 'Tecplot mesh_topography data has been read.'

    do i=1, numNodes
	  read (2,*) nNode(i), x(nNode(i)), y(nNode(i))  ! 节点编号 节点的x,y坐标
	enddo

! write to hgrid.gr3 file
     do i=1, numNodes
       write(3,9) nNode(i), x(nNode(i)), y(nNode(i)),z(nNode(i)) 
	 enddo
9 format(I6,1x,2(F15.3,1x),F10.3)

	  read(2,*) !dataTemp   ! ENDOFSECTION 
      read(2,*) !dataTemp   ! ELEMENTS/CELLS 2.4.6

	do i=1, numEles
	!           单元编号  3        单元类型    单元的3个节点编号   
	   read (2,*) nEle(i),NumElep1,numElep2,(nNodes(k,nEle(i)),k=1,numElep2)
    enddo


	do i=1, numEles
       write(3,*) nEle(i),numElep2,(nNodes(k,nEle(i)),k=1,numElep2)
	enddo

 write(*,*) 'the data in neu file has been read'

	  read(2,*) !dataTemp   ! ENDOFSECTION 
! 把group删除
! 读取边界信息及边界上的节点号
! cj-inlet 
      read(2,*)  ! BOUNDARY CONDITIONS 2.4.6
      read(2,*) inlet_name1,temp1, inlet_node_numbers1,temp2,temp3     
      write(*,'(a10,1x,I5)') inlet_name1,inlet_node_numbers1
      write(4,'(a10,1x,I5)') inlet_name1,inlet_node_numbers1
      do i=1,inlet_node_numbers1
         read(2,*) inlet_node1(i)
	  enddo
      read(2,*) ! ENDOFSECTION 
!  jialingjiang - inlet 
      read(2,*)  ! BOUNDARY CONDITIONS 2.4.6
      read(2,*) inlet_name2,temp1, inlet_node_numbers2,temp2,temp3     
      write(*,'(a10,1x,I5)') inlet_name2,inlet_node_numbers2
      write(4,'(a10,1x,I5)') inlet_name2,inlet_node_numbers2
      do i=1,inlet_node_numbers2
         read(2,*) inlet_node2(i)
	  enddo
      read(2,*) ! ENDOFSECTION 
! wujiang- inlet
      read(2,*)  ! BOUNDARY CONDITIONS 2.4.6
      read(2,*) inlet_name3,temp1, inlet_node_numbers3,temp2,temp3     
      write(*,'(a10,1x,I5)') inlet_name3,inlet_node_numbers3
      write(4,'(a10,1x,I5)') inlet_name3,inlet_node_numbers3
      do i=1,inlet_node_numbers3
         read(2,*) inlet_node3(i)
	  enddo
      read(2,*) ! ENDOFSECTION 
! outlet  - dam
      read(2,*)
      read(2,*) outlet_name,temp1, outlet_node_numbers,temp2,temp3     
      write(*,'(a10,1x,I5)') outlet_name,outlet_node_numbers
      write(4,'(a10,1x,I5)') outlet_name,outlet_node_numbers
      do i=1,outlet_node_numbers
         read(2,*) outlet_node(i)
	  enddo
      read(2,*)
!------------------------------------------------------
! left_bank from cj to jianglingjiang
      read(2,*)
      read(2,*) left_bank_name1,temp1, left_bank_node_numbers1,temp2,temp3     
      write(*,'(a10,1x,I5)') left_bank_name1,left_bank_node_numbers1
      write(4,'(a10,1x,I5)') left_bank_name1,left_bank_node_numbers1
      do i=1,left_bank_node_numbers1
         read(2,*) left_bank_node1(i)
	  enddo
      read(2,*)

! left_bank from jianglingjiang to dam
      read(2,*)
      read(2,*) left_bank_name2,temp1, left_bank_node_numbers2,temp2,temp3     
      write(*,'(a10,1x,I5)') left_bank_name2,left_bank_node_numbers2
      write(4,'(a10,1x,I5)') left_bank_name2,left_bank_node_numbers2
      do i=1,left_bank_node_numbers2
         read(2,*) left_bank_node2(i)
	  enddo
      read(2,*)

! right_bank from cj to wujiang--1
      read(2,*)
      read(2,*) right_bank_name1,temp1, right_bank_node_numbers1,temp2,temp3     
      write(*,'(a10,1x,I5)') right_bank_name1,right_bank_node_numbers1
      write(4,'(a10,1x,I5)') right_bank_name1,right_bank_node_numbers1
      do i=1,right_bank_node_numbers1
         read(2,*) right_bank_node1(i)
	  enddo
      read(2,*)
! right_bank from cj to wujiang--2
      read(2,*)
      read(2,*) right_bank_name2,temp1, right_bank_node_numbers2,temp2,temp3     
      write(*,'(a10,1x,I5)') right_bank_name2,right_bank_node_numbers2
      write(4,'(a10,1x,I5)') right_bank_name2,right_bank_node_numbers2
      do i=1,right_bank_node_numbers2
         read(2,*) right_bank_node2(i)
	  enddo
      read(2,*)
! right bank from wujiang to dam
      read(2,*)
      read(2,*) right_bank_name3,temp1, right_bank_node_numbers3,temp2,temp3     
      write(*,'(a10,1x,I5)') right_bank_name3,right_bank_node_numbers3
      write(4,'(a10,1x,I5)') right_bank_name3,right_bank_node_numbers3
      do i=1,right_bank_node_numbers3
         read(2,*) right_bank_node3(i)
	  enddo
      read(2,*)

! islands
do l=1,9
      read(2,*) 
      read(2,*) island_name(l),temp1, island_node_numbers(l),temp2,temp3     
      write(*,'(a10,1x,I5)') island_name(l),island_node_numbers(l)
      write(4,'(a10,1x,I5)') island_name(l),island_node_numbers(l)
      do i=1,island_node_numbers(l)
         read(2,*) island_node(l,i)
	  enddo
      read(2,*)  ! ENDOFSECTION 
enddo
close(4)
!
!------------------------------------------------------------
! output boundary condition to hgrid.gr3 file
!------------------------------------------------------------
write(3,*) 4,'=Number of open boundary'
write(3,*) inlet_node_numbers1+inlet_node_numbers2+inlet_node_numbers3+outlet_node_numbers, &
         '=Total number of oen boundary'
write(3,*) inlet_node_numbers1,'=Number of open boundary 1 - cj inlet'

do i=1,inlet_node_numbers1
  write(3,*) inlet_node1(i) 
enddo	
 
 write(*,*) 'cj inlet boundary node outputs.'


write(3,*) inlet_node_numbers2,'=Number of open boundary 2 - jlj inlet'
! jialingjiang inlet
 do i=1,inlet_node_numbers2
  write(3,*) inlet_node2(i) 
enddo

 write(*,*) 'jialingjiang inlet boundary node outputs.'

write(3,*) inlet_node_numbers3,'=Number of open boundary 3 - wj inlet'
! wujiang inlet
  do i=1,inlet_node_numbers3
  write(3,*) inlet_node3(i) 
  enddo
 write(*,*) 'wujiang inlet boundary node outputs.'

!
write(3,*) outlet_node_numbers,'=Number of open boundary 4 - dam outlet'
  do i=1,outlet_node_numbers
  write(3,*) outlet_node(i) 
  enddo

write(*,*) 'dam outlet boundary node outputs.'


!-----------------land boundary node outputs-----------------------------
write(3,*) 5+9,'=number of land boundary(including islands)'
m=0
do l=1,9
m=m+island_node_numbers(l)
enddo

write(3,*) left_bank_node_numbers1+left_bank_node_numbers2+  &
right_bank_node_numbers1+right_bank_node_numbers2+right_bank_node_numbers3+m,  &
'=Total bumber of land boundary nodes'

! left bank from cj to jlj
write(3,*) left_bank_node_numbers1,0,'=Number of land boundary 1 - left bank cj2jlj'
do i=1,left_bank_node_numbers1
  write(3,*) left_bank_node1(i) 
enddo
write(*,*) 'cj2jlj left bank boundary node outputs.'

! left bank from jlj to dam
write(3,*) left_bank_node_numbers2,0,'=Number of land boundary 2 - left bank jlj2dam'
do i=1,left_bank_node_numbers2
  write(3,*) left_bank_node2(i) 
enddo
write(*,*) 'jlj2dam left bank boundary node outputs.'


!
write(3,*) right_bank_node_numbers1,0,'=Number of land boundary 3 - right bank cj2wj'
do i=1,right_bank_node_numbers1
  write(3,*) right_bank_node1(i) 
enddo

 write(*,*) 'cj2wj1 right bank boundary node outputs.'


!
write(3,*) right_bank_node_numbers2,0,'=Number of land boundary 4 - right bank cj2wj'
do i=1,right_bank_node_numbers2
  write(3,*) right_bank_node2(i) 
enddo
 write(*,*) 'cj2wj1-2 right bank boundary node outputs.'


! wujiang to dam
write(3,*) right_bank_node_numbers3,0,'=Number of land boundary 5 - right bank wj2dam'
do i=1,right_bank_node_numbers3
  write(3,*) right_bank_node3(i) 
enddo
 write(*,*) 'wj2dam right bank boundary node outputs.'


do m=1,9
! islands
write(3,*) island_node_numbers(m),1,'=Number of land boundary - island' 
do i=1,island_node_numbers(m)
   write(3,*) island_node(m,i) 
enddo
write(*,*) 'hhc island boundary node outputs.',m
enddo






     close(3)

     print *, 'Voila!'

     end

	  
 
