
!!!!!
program geo

include 'variable.cmn'
integer in_ele(kMesh),out_ele(kMesh),facein(kMesh,2),faceout(kMesh,2)
real*8 LP1f,La1f,Lai,nx,ny

cta=cta0/180.0*4.0d0*datan(1.0d0)
eps=1d-10
!==========================information of node======================================
!open(1,file=files)
open(1,file='test.neu',status='old')
do i=1,9
	read(1,*)
end do

do i=1,kNode
	read(1,*)ii,fNodeX(i),fNodeY(i)
fNodeX(i)=1*fNodeX(i)
fNodeY(i)=1*fNodeY(i)
end do

read(1,*)
read(1,*)
do i=1,Mesh
	read(1,*)ii,id1,id2,(iMeshNode(i,j),j=1,id2) !逆时针方向
	if(id2==3)iMeshNode(i,4)=iMeshNode(i,1)             
end do



close(1)
!===============boundary information of inlet and outlet===========================
!open(2,file='in_out.txt')
!read(2,*)
!do i=1,incount
!   read(2,*)in_ele(i),facein(i,1),facein(i,2)
!end do

!read(2,*)
!do i=1,outcount
!   read(2,*)out_ele(i),faceout(i,1),faceout(i,2)
!end do
!close(2)

!===============search the elements contain the same node ===============
ii=0
do j=1,kNode
!if(mod(j,100).eq.0)write(*,*)j
	do i=1,kMesh   
		i1=iMeshNode(i,1);i2=iMeshNode(i,2);i3=iMeshNode(i,3);i4=iMeshNode(i,4)
		if(i1.eq.i4)then !triangle
		  nside(i)=3
		  if(i1.eq.j.or.i2.eq.j.or.i3.eq.j)then
		  ii=ii+1
		  node_nei_ele(j,ii)=i
          node_nei_ele_count(j)=ii
		  end if
		else             !quadrilateral
		  nside(i)=4
		  if(i1.eq.j.or.i2.eq.j.or.i3.eq.j.or.j.eq.i4)then
		  ii=ii+1
		  node_nei_ele(j,ii)=i
          node_nei_ele_count(j)=ii
		  end if
		end if
	enddo
  ii=0
end do
!========================确定该单元的临近单元=================================================
!face1=========inlet boundary-1
!face2=========outlet boundary-2
!face3=========wall boundary 0
epss=1d-3
do i=1,Mesh
!if(mod(i,100).eq.0)write(*,*)i
    do 800 j=1,nside(i)
	  j1=j+1
	  if(j.eq.nside(i))j1=1
	  i1=iMeshNode(i,j);i2=iMeshNode(i,j1)
!======================input boundary search===============================
	     kk=0  			  
		 if(fNodeX(i1).le.6.786+0.003.and.fNodeX(i1).ge.6.716-0.003.and.abs(fNodeY(i1)).le.0.001)kk=kk+1
		 if(fNodeX(i2).le.6.786+0.003.and.fNodeX(i2).ge.6.716-0.003.and.abs(fNodeY(i2)).le.0.001)kk=kk+1		       
	     if(kk==2)then
		   iMeshNei(i,j)=-1
		   go to 800
         end if	
!======================output boundary search==============================
	     kk=0  
		 if(fNodeX(i1).le.2.702+0.003.and.fNodeX(i1).ge.2.632-0.003.and.abs(fNodeY(i1)).le.0.001)kk=kk+1
		 if(fNodeX(i2).le.2.702+0.003.and.fNodeX(i2).ge.2.632-0.003.and.abs(fNodeY(i2)).le.0.001)kk=kk+1
	     if(kk==2)then
		   iMeshNei(i,j)=-2
		   go to 800
          end if	
!======================input boundary search===============================
		 do if1=1,incount
		  j1=facein(if1,1);j2=facein(if1,2)
		    if( (i1.eq.j1.and.i2.eq.j2).or.(i1.eq.j2.and.i2.eq.j1))then
		  		  !iMeshNei(i,j)=-1 !input boundary
		    	  !goto 800
		    end if
		 end do
!======================output boundary search==============================
	     do if1=1,outcount
		 j1=faceout(if1,1);j2=faceout(if1,2)
		   if( (i1.eq.j1.and.i2.eq.j2).or.(i1.eq.j2.and.i2.eq.j1))then
				  !iMeshNei(i,j)=-2 !output boundary
			      !goto 800
		   end if
		 end do
!=====================common face search===========================================================
        do kk=1,2
		 if(kk.eq.1)i4=i1
		 if(kk.eq.2)i4=i2
		  do  jj0=1,node_nei_ele_count(i4) !total number of elements containing the node i1
			  ii=node_nei_ele(i4,jj0)      !the element 
		    	if(i.ne.ii)then
				   do jj=1,nside(ii) 
				    jj1=jj+1
					if(jj.eq.nside(ii))jj1=1
					j1=iMeshNode(ii,jj);j2=iMeshNode(ii,jj1)
					  if( (i1.eq.j1.and.i2.eq.j2).or.(i1.eq.j2.and.i2.eq.j1))then
						iMeshNei(i,j)=ii   !common face 
						goto 800
					  end if
				  end do
				end if		
          end do
	   end do
!===================================================================================================
800 continue
enddo


!=================================matrix transfering==============================================
kk=0
do i=1,Mesh
   do j=1,nside(i)
     iMeshNei0(i,j)=0

if(iMeshNei(i,j).gt.0)then
   iMeshNei0(i,j)=iMeshNei(i,j)

else if(iMeshNei(i,j).eq.0)then !solid boundary 
   kk=kk+1
   iMeshNei0(i,j)=iMeshNei0(i,j)+Mesh+kk
else if(iMeshNei(i,j).eq.-1)then !input boundary 
   kk=kk+1
   iMeshNei0(i,j)=iMeshNei0(i,j)+Mesh+kk
else if(iMeshNei(i,j).eq.-2)then !output boundary
   kk=kk+1 
   iMeshNei0(i,j)=iMeshNei0(i,j)+Mesh+kk
else


endif
enddo
enddo

! iTolMesh=Mesh+kk
! write(*,*)iTolMesh

stop
end 
