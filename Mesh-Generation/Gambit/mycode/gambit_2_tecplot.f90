  program main
	implicit none
	real,parameter :: PAI=3.1415926
	character :: filename*50
	character :: dataTemp*100
	integer :: indexFile(2)
	integer :: temp1,temp2,temp3,temp4
	integer :: m,numnodes,numeles,numelep1,numelep2
	integer :: i,k,mm
	real :: ze,zeet
	real,allocatable:: x(:),y(:),z(:),zee(:),x1(:),y1(:)
	real,external :: zb1,zb
	integer,allocatable::nNodes(:,:),nEle(:),nNode(:)
	
    indexFile(1)=1   ! 输出tecplot格式的开关
	indexFile(2)=0   ! 输出FVM2D格式的开关

!Gambit导出的网格文件  Polyflow求解器格式
!	filename='confluence.neu'
	write(*,*)'输入neu文件：*.neu'
	read(*,*) filename
	open(unit=2,file=filename,status='old')

	write(*,*)'输出Tecplot文件：*.dat'
	read(*,*)filename
	open(unit=3,file=filename,status='replace')
	
!---------- Read from GAMBIT neutral file ----------------------
      do m=1,9
	  if (m.ne.7) then
          read(2,*) !dataTemp
	  else
          read(2,*) numNodes, numEles  !,temp1,temp2,temp3,temp4
	    allocate(x(numNodes),y(numNodes),z(numNodes),nNode(numNodes),  &
		nNodes(3,numEles),nEle(numEles),zee(numEles))   	     
		allocate(x1(numNodes),y1(numNodes))
	  endif
	enddo

    do i=1, numNodes
	  read (2,*) nNode(i), x(nNode(i)), y(nNode(i)) !,z(nNode(i))
!     ze=zb( x(nNode(i)), y(nNode(i)))
!     z(nNode(i))=ze
!	  write(*,*) nNode(i), x(nNode(i)), y(nNode(i)), z(nNode(i))
	enddo
	do m=1,2
	  read(2,*) dataTemp
	enddo
	do 900 i=1, numEles
	   read (2,*) nEle(i),NumElep1,numElep2,(nNodes(k,nEle(i)),k=1,numElep2)
        zeet=0.0
	  do 600 mm=1,numElep2
	      zeet=zeet+z(nNodes(mm,nEle(i)))
  600	  enddo
	  zee(nEle(i))=zeet/numElep2     
  900 enddo

!--------- TO Tecplot -------------------------  
	write(*,*) 'Nodes:',numNodes, 'Elements:',numEles
    if (indexFile(1).eq.1) then
    write(3,*) 'TITLE     = ""  '
	write(3,*) 'VARIABLES = "x"'
	write(3,*) '"y"'
	write(3,*) '"z"'
      if (numElep2.eq.3) then  ! trianglar element
      write(3,*)'ZONE T="triangle"'
	write(3,*) 'N=',numNodes,',E=',numEles,',ZONETYPE=FETRIANGLE'
	elseif (numElep2.eq.4) then      ! quadrature element
	write(3,*)'ZONE T="quadrangle"'
	write(3,*) 'N=',numNodes,',E=',numEles,',ZONETYPE=FEQuadrilateral'
	endif
	write(3,*) 'DATAPACKING=POINT'
	write(3,*) 'DT=(SINGLE SINGLE SINGLE )'
	do i=1, numNodes
        write(3,101) x(nNode(i)), y(nNode(i)), z(nNode(i))
	enddo
	do 90 i=1, numEles
        write(3,*)(nNodes(k,nEle(i)),k=1,numElep2)      
   90 enddo
      endif

! 导出为FVM2D程序输入文件格式（张大伟师兄的程序）
	filename='FVM2D.dat'
	open(unit=6,file=filename,status='replace')	  
      if (indexFile(2).eq.1) then
      write(6,*) numNodes, numEles 
	do i=1,numNodes
      write(6,102) nNode(i),	x(nNode(i)), y(nNode(i))
	enddo
	if (numElep2.eq.3) then
	do i=1,numEles
!      zee(nEle(i))=0.0
      write(6,1036)nEle(i),(nNodes(k,nEle(i)),k=1,numElep2),zee(nEle(i))
	enddo
	elseif (numElep2.eq.4) then
	do i=1,numEles
      write(6,1032)nEle(i),(nNodes(k,nEle(i)),k=1,numElep2),zee(nEle(i))
	enddo
      endif
	endif
  101 format(22e20.8)
  102 format(1x,i10,22e20.8,22e20.8)      
 1032 format (1x,i10,i10,i10,i10,i10,22e20.8)
 1036 format (1x,i10,i10,i10,i10,22e20.8)
      print *, 'ok!'
	  
	  close(2)	  
	  close(3)
	  close(6)
	  stop
	end
	
	
	
!
! 一些自定义的处理地形高程数据的函数
!
!	do i=1, numNodes   !直角坐标系坐标旋转，逆时针90度
!	   x(nNode(i))=!x(nNode(i))*COS(-PAI/2)+y(nNode(i))*SIN(-PAI/2)
!	   y(nNode(i))=!x(nNode(i))*SIN(-PAI/2)+y(nNode(i))*COS(-PAI/2)
!	enddo

!
	real function zb1(xt,yt)
	implicit none
    real xt,yt,zt,xxt1,xxt2,xxt3
	zb1=0.0
	if (xt.gt.8.0.and.xt.lt.12.0) then
	    zb1=0.2-0.05*(xt-10.)**2
	    return
	endif
	return
	end function	  
        
	real function zb(xt,yt)
	implicit none
      real xt,yt,zt,xxt1,xxt2,xxt3
	zb=0.0
	xxt1=(xt-30.)**2+(yt-8.)**2
	if (xxt1.le.42.25) then
	    zb=1-((xt-30.)**2+(yt-8.)**2)/42.25
	    return
	endif
      xxt2=(xt-30.)**2+(yt-22.)**2
      if (xxt2.le.42.25) then
	    zb=1-((xt-30.)**2+(yt-22.)**2)/42.25
	    return
	endif
      xxt3=(xt-52.)**2+(yt-15.)**2
	if (xxt3.le.73.96) then
	    zb=2-((xt-52.)**2+(yt-15.)**2)/36.98
	    return
	endif
	return	
	end function