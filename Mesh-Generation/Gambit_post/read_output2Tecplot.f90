!===============================================================================
! Read binary format 3.0 (V5.01)
! *** MODIFIED FOR OUTPUT FROM PARALLEL CODE ***
! Same as read_output4.f but allows for reading multiple files for a node.
!===============================================================================

program read_out
!-------------------------------------------------------------------------------
  implicit real(4)(a-h,o-z),integer(i-n)
  parameter(nbyte=4)
  character*30 file63,file65
  character*12 it_char
  character*48 start_time,version,variable_nm,variable_dim
  character*48 data_format
  allocatable ztot(:),x(:),y(:),dp(:),kbp(:),kfp(:)
  allocatable i34(:),nm(:,:),outb(:,:)
!-------------------------------------------------------------------------------

!-------------------------------------------------------------------------------
! Aquire user inputs
!-------------------------------------------------------------------------------

  write(*,'(a)',advance='no') 'Input file to read from (without *_): '  !输入 需要读取的文件名
  read(*,'(a30)') file63
      
  write(*,'(a)',advance='no') 'Input initial and final file index: '   !输入 初始和最终的文件标号
  read(*,*) ibgn,iend

 ! write(*,'(a)',advance='no') 'Input node number: '      !输入节点号
 ! read(*,*) node

  write(*,'(a)',advance='no') 'Input output file name: '   !输入 输出文件名
  read(*,'(a30)') file65
      
! Open output file
  open(65,file=file65,status='replace')

!-------------------------------------------------------------------------------
! Scan header data of first file and set some parameters
!-------------------------------------------------------------------------------

! Header
  write(it_char,'(i12)') ibgn
  it_char=adjustl(it_char)  !place blanks at end
  it_len=len_trim(it_char)  !length without trailing blanks
  open(63,file=it_char(1:it_len)//'_'//file63,status='old',access='direct',recl=nbyte)
  irec=0
  do m=1,48/nbyte
    read(63,rec=irec+m) data_format(nbyte*(m-1)+1:nbyte*m)
  enddo
  if(data_format.ne.'DataFormat v3.0') then
    write(*,*)'Unknown data format',data_format
    stop
  endif
  irec=irec+48/nbyte
  do m=1,48/nbyte
    read(63,rec=irec+m) version(nbyte*(m-1)+1:nbyte*m)
  enddo
  irec=irec+48/nbyte
  do m=1,48/nbyte
    read(63,rec=irec+m) start_time(nbyte*(m-1)+1:nbyte*m)
  enddo
  irec=irec+48/nbyte
  do m=1,48/nbyte
    read(63,rec=irec+m) variable_nm(nbyte*(m-1)+1:nbyte*m)
  enddo
  irec=irec+48/nbyte
  do m=1,48/nbyte
    read(63,rec=irec+m) variable_dim(nbyte*(m-1)+1:nbyte*m)
  enddo
  irec=irec+48/nbyte

 ! write(65,'(a,a48)') '# ',data_format
 ! write(65,'(a,a48)') '# ',version
 ! write(65,'(a,a48)') '# ',start_time
 ! write(65,'(a,a48)') '# ',variable_nm
 ! write(65,'(a,a48)') '# ',variable_dim

  read(63,rec=irec+1) nrec
  irec_nrec=irec+1
  read(63,rec=irec+2) dtout
  read(63,rec=irec+3) nspool
  read(63,rec=irec+4) ivs
  read(63,rec=irec+5) i23d
  read(63,rec=irec+6) vpos
  irec=irec+6

  !print*, 'nrec=',nrec,'; dtout=',dtout,'; nspool=',nspool
  !print*, 'ivs=',ivs,'; i23d=',i23d,'; vpos=',vpos

! Vertical grid
  read(63,rec=irec+1) zmsl
  read(63,rec=irec+2) nvrt
  irec=irec+2
  allocate(ztot(nvrt),outb(0:nvrt,2))
  do k=1,nvrt
    read(63,rec=irec+k) ztot(k)
  enddo
  irec=irec+nvrt

!print*,'ZMSL,NVRT: ',zmsl,nvrt

! Horizontal grid size
  read(63,rec=irec+1) np
  read(63,rec=irec+2) ne
  irec=irec+2


! Allocate horizontal grid data
  allocate(x(np),y(np),dp(np),kbp(np),kfp(np))
  allocate(i34(ne),nm(ne,4))

! 节点数据
  do m=1,np
    read(63,rec=irec+1)x(m)
    read(63,rec=irec+2)y(m)
    read(63,rec=irec+3)dp(m)
    read(63,rec=irec+4)kbp(m)
    !write(*,'(a,2i8,3e14.6,i4)')'NODE: ',m,irec,x(m),y(m),dp(m),kbp(m)
    irec=irec+4
  enddo !m=1,np

! 单元数据
  do m=1,ne
    read(63,rec=irec+1)i34(m)
    do mm=1,i34(m)
      read(63,rec=irec+1+mm)nm(m,mm)
    enddo !mm
    !write(*,'(a,2i8,i3,4i8)')'ELEM: ',m,irec,i34(m),(nm(m,j),j=1,i34(m))
    irec=irec+5
  enddo !m

! Last record of header in each file
  irec0=irec

! Multiplier for node-record increment
  if(i23d.eq.2) then !2D
    if(ivs.eq.1) then
      mrec=1+1
    else
      mrec=1+2
    endif
  else !3D
    if(ivs.eq.1) then
      mrec=1+  nvrt
    else
      mrec=1+2*nvrt
    endif
  endif

! Close before time iteration
  close(63)

!-------------------------------------------------------------------------------
! Time iteration -- select "node" data
!-------------------------------------------------------------------------------

! Loop over input files
  do iinput=ibgn,iend

    ! Open input file
    write(it_char,'(i12)')iinput
    it_char=adjustl(it_char)  !place blanks at end
    it_len=len_trim(it_char)  !length without trailing blanks
    open(63,file=it_char(1:it_len)//'_'//file63,status='old',access='direct',recl=nbyte)

    ! Read actual number of spools in this file
    read(63,rec=irec_nrec) nrec

    ! Loop over output spools in file
    do ispool=1,nrec

  !print*,'NP,NE: ',np,ne   Tecplot header
  if(i23d==2) then
     write(65,*) 'VARIABLES="X","Y","Parameter"'
     write(65,*) 'ZONE N=',NP,'  E=',NE,'  F=FEPOINT,    ET=Quadrilateral'
  else !i23d==3
     write(65,*) 'VARIABLES="X","Y","Z","Parameter1","Parameter2"'
	 write(65,*) 'ZONE T="VOLUME",','  N=', np*nvrt,',E='  ,ne*(nvrt-1), '  ,F=FEPOINT,ET=BRICK'   
 endif

       do node=1,np
      ! Starting record for data
      irec=irec0+(ispool-1)*(2+mrec*np)

      read(63,rec=irec+1) time
      read(63,rec=irec+2) it
      irec=irec+2

 !     print*, 'it=',it,' time=',time/86400.0
       
      node_rec=irec+mrec*(node-1)+1 !skip to node record

      read(63,rec=node_rec) kfp(node)     !目前只能输出表层的平面2维数据

      if(i23d.eq.2) then  !2-dimensional
        do m=1,ivs
          read(63,rec=node_rec+m) outb(0,m) 
        enddo !m
 !      write(65,'(500e14.6)')time/86400.0,(outb(0,m),m=1,ivs)  !ASCII
        write(65,10) x(node),y(node),(outb(0,m),m=1,ivs)        !ASCII
10   format(2(1x,f15.3),1x,f15.6)

      else !i23d=3  3-dimensional
        do k=kbp(node),kfp(node)
          do m=1,ivs
            read(63,rec=node_rec+ivs*(k-1)+m) outb(k,m)
          enddo !m
        enddo !k
 !       write(65,'(I8,2(1x,i4),1x,50(1x,e12.3))')node,kbp(node),kfp(node),&
 !       ((outb(k,m),m=1,ivs),k=kbp(node),kfp(node))    !ASCII

      endif !i23d
     enddo !node=1,np
     
    if(i23d.eq.2) then  !2-dimensional
	   do m=1,ne
          write(65,112) (nm(m,j),j=1,i34(m))
	   enddo
	else  !3D
!... Extend  for wet nodes  湿节点计算值扩展到整个垂向网格上
do k=1,kbp(node)-1
  do m=1,ivs
    outb(k,m)=outb(kbp(node),m)
  enddo
enddo

do k=kfp(node)+1,nvrt
  do m=1,ivs
    outb(k,m)=outb(kfp(node),m)
  enddo
enddo

  do k=1,nvrt	
	do i=1,np
      write(65,113) x(i),y(i),ztot(k),(outb(k,m),m=1,ivs)
	enddo
 enddo

	do k=1,nvrt-1
    do i=1,ne			   		
		write(65,115)(nm(i,j)+(k-1)*np,j=1,i34(i)),(nm(i,j)+k*np,j=1,i34(i))
	enddo
	enddo
	endif
112  format(4(1x,I8))
113 format(3(1x,f15.3),2(1x,f15.6))
115 format(8(1x,I10))
    enddo !ispool=1,nrec

    ! Close input file
    close(63)

   ! Close output file
    close(65)
  enddo !iinput=1,ninput_files
stop
end program read_out
