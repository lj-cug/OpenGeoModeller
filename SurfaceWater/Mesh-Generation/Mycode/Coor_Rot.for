	program xuanzhuan
	parameter(numnodes=22695,numeles=21476)
	
	character filename*50
	character dataTemp*100

      dimension x(numnodes),y(numnodes),z(numnodes)
     &        ,x1(numnodes),y1(numnodes)
	dimension nep(numeles,4)


	filename='XX_dixing.dat'
	open(unit=2,file=filename,status='old')
      filename='XXR_TEC_X.dat'
	open(unit=3,file=filename)
	filename='XXDixing.dat'

	do i=1,8
	read(2,*)
	enddo

	do i=1,numnodes
	read(2,*)x(i),y(i),z(i)
	enddo

	do i=1,numeles
	read(2,*)(nep(i,j),j=1,4)
	enddo

	do i=1, numNodes
	x1(i)=x(i)
	y1(i)=y(i)
	enddo
	
	do i=1,numNodes   !直角坐标系坐标旋转，逆时针90度
	   x(i)=-y1(i) !x(nNode(i))*COS(-PAI/2)+y(nNode(i))*SIN(-PAI/2)
	   y(i)=x1(i)  !x(nNode(i))*SIN(-PAI/2)+y(nNode(i))*COS(-PAI/2)
	enddo

	do i=1,numnodes
	   write(3,10)x(i),y(i),z(i)
	enddo
10	format(1x,f20.3,1x,f20.3,1x,f10.5)
	do i=1,numeles
	write(3,*)(nep(i,j),j=1,4)
	enddo
	stop
	end