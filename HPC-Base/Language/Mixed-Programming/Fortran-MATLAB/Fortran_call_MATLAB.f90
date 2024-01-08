! include the path of C:\MATLAB\extern\include
! Add the path of extern lib and the static libraries:
! libmat.lib libeng.lib libmx.lib
!
program main

implicit none
!-------------------
! �����64λ�Ļ����ͳ��������ⲿ����Ҫ����Ϊinteger*8����
integer*4,external :: engOpen,engClose,mxCreateDoubleMatrix
integer*4,external :: mxGetPr,engPutVariable,engEvalString
integer*4 :: ep,T   ! 32λ����ָ��
integer :: status
!----------------------------

integer :: i,j
real*8 :: array(20,30)

forall(i=1:20,j=1:30)
array(i,j)=i*j+j+(20-i)*(30-j)
end forall

! ��MATLAB����
ep = engOpen('')
if(ep==0) then
  write(6,*)'Cannot start MATLAB engine.'
  stop
endif

! ��������
T=mxCreateDoubleMatrix(20,30,0)
call mxCopyReal8ToPtr(array,mxGetPr(T),20*30)
status=engPutVariable(ep,'T',T)
if(status/=0) then
  write(*,*)'engPutVariable failed!'
  stop
endif

! �����ͼָ��
if(engEvalString(ep,'pcolor(T);axis ij;')/=0) then
!                    �˴������MATLAB��ָ����ȫһ��
  write(*,*)'engEvalString failed!'
  stop
endif

! �ر�����
write(*,*)'Please press[Enter] to continue'
read(*,*)
call mxDestroyArray(T)
status=engClose(ep)
if(status/=0) then
  write(*,*)'engClose failed!'
  stop
endif

end