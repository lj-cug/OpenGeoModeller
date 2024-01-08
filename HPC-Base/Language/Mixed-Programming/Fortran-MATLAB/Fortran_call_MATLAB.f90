! include the path of C:\MATLAB\extern\include
! Add the path of extern lib and the static libraries:
! libmat.lib libeng.lib libmx.lib
!
program main

implicit none
!-------------------
! 如果是64位的机器和程序，下列外部函数要定义为integer*8类型
integer*4,external :: engOpen,engClose,mxCreateDoubleMatrix
integer*4,external :: mxGetPr,engPutVariable,engEvalString
integer*4 :: ep,T   ! 32位整型指针
integer :: status
!----------------------------

integer :: i,j
real*8 :: array(20,30)

forall(i=1:20,j=1:30)
array(i,j)=i*j+j+(20-i)*(30-j)
end forall

! 打开MATLAB引擎
ep = engOpen('')
if(ep==0) then
  write(6,*)'Cannot start MATLAB engine.'
  stop
endif

! 传送数组
T=mxCreateDoubleMatrix(20,30,0)
call mxCopyReal8ToPtr(array,mxGetPr(T),20*30)
status=engPutVariable(ep,'T',T)
if(status/=0) then
  write(*,*)'engPutVariable failed!'
  stop
endif

! 传达绘图指令
if(engEvalString(ep,'pcolor(T);axis ij;')/=0) then
!                    此处命令跟MATLAB的指令完全一样
  write(*,*)'engEvalString failed!'
  stop
endif

! 关闭引擎
write(*,*)'Please press[Enter] to continue'
read(*,*)
call mxDestroyArray(T)
status=engClose(ep)
if(status/=0) then
  write(*,*)'engClose failed!'
  stop
endif

end