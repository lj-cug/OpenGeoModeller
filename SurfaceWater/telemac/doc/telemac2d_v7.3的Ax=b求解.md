AX=B 线性方程组求解器

以propag.F为例：Line 1633 and Line 1655
(分别求解原始方程和统一的波动方程)

[CALL SOLVE(UNK,MAT,RHS,TB,SLVPRO,INFOGR,MESH,TM1)]{.mark}

[!\| UNK \|\<-\>\| BLOCK OF UNKNOWNS]{.mark} X

[!\| MAT \|\<\--\| BLOCK OF MATRICES]{.mark} A

[!\| RHS \|\<-\>\| BLOCK OF PRIVATE BIEF_OBJ STRUCTURES]{.mark} B

[!\| TB \|\<-\>\| BLOCK WITH T1,T2,\...]{.mark}

[!\| SLVPRO \|\--\>\| SOLVER STRUCTURE FOR PROPAGATION]{.mark}

[!\| INFOGR \|\--\>\| IF YES, INFORMATION ON GRADIENT]{.mark}

[!\| MESH \|\--\>\| MESH STRUCTURE]{.mark}

[!\| TM1 \|\<-\>\| MATRIX]{.mark}

使用共轭梯度法时，需要的输入变量：[UNK,MAT,RHS]{.mark}

[!
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*]{.mark}

[SUBROUTINE SOLVE (X, A, B,TB,CFG,INFOGR,MESH,AUX)]{.mark}

[!
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*]{.mark}

[!\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~]{.mark}

[!\| A \|\--\>\| MATRIX OF THE SYSTEM (OR BLOCK OF MATRICES)]{.mark}

[!\| AUX \|\--\>\| MATRIX FOR PRECONDITIONING.]{.mark}

[!\| B \|\--\>\| RIGHT-HAND SIDE OF THE SYSTEM]{.mark}

[!\| CFG \|\--\>\| STRUCTURE OF SOLVER CONFIGURATION]{.mark}

[!\| \| \| CFG%KRYLOV IS USED ONLY IF CFG%SLV = 7 (GMRES)]{.mark}

[!\| INFOGR \|\--\>\| IF YES, PRINT A LOG.]{.mark}

[!\| MESH \|\--\>\| MESH STRUCTURE.]{.mark}

[!\| TB \|\--\>\| BLOCK OF VECTORS WITh AT LEAST]{.mark}

[!\| \| \| MAX(7,2+2\*CFG%KRYLOV)\*S VECTORS, S IS 1]{.mark}

[!\| \| \| IF A IS A MATRIX, 2 IF A BLOCK OF 4 MATRICES]{.mark}

[!\| \| \| AND 3 IF A BLOCK OF 9.]{.mark}

[!\| X \|\<-\>\| INITIAL VALUE, THEN SOLUTION]{.mark}

[TYPE(SLVCFG), INTENT(INOUT) :: CFG]{.mark}

[!]{.mark}

[! STRUCTURES OF VECTORS OR BLOCKS OF VECTORS]{.mark}

[!]{.mark}

[TYPE(BIEF_OBJ), TARGET, INTENT(INOUT) :: X,B]{.mark}

[TYPE(BIEF_OBJ), INTENT(INOUT) :: TB]{.mark}

[!]{.mark}

[! STRUCTURES OF MATRIX OR BLOCK OF MATRICES]{.mark}

[!]{.mark}

[TYPE(BIEF_OBJ), INTENT(INOUT) :: A, AUX]{.mark}

[!]{.mark}

[LOGICAL, INTENT(IN) :: INFOGR]{.mark}

[!]{.mark}

[! MESH STRUCTURE]{.mark}

[!]{.mark}

[ TYPE(BIEF_MESH), INTENT(INOUT) :: MESH]{.mark}

[对角矩阵预处理：]{.mark}

[CALL PRECDT(X,A,B,TBB%ADR(IT1)%P,MESH,
CFG%PRECON,PREXSM,DIADON,S)]{.mark}

[ PRECD1(X, A, B, D%ADR(1)%P, MESH, PRECON, PREXSM, DIADON)]{.mark}

[!]{.mark}

[! CONJUGATE GRADIENT]{.mark}

[!]{.mark}

[CALL GRACJG(PX, A, PB, MESH,]{.mark}

[& TBB%ADR(IT2)%P,TBB%ADR(IT3)%P,]{.mark}

[& TBB%ADR(IT5)%P,TBB%ADR(IG)%P,]{.mark}

[& CFG,INFOGR,AUX)]{.mark}

YSMP直接求解器不能并行计算，对GPU加速的求解器有启示。模仿来编程！

实参：

[CALL SD_SOLVE_1(A%D%DIM1,MESH%NSEG,MESH%GLOSEG%I,]{.mark}

[& MESH%GLOSEG%DIM1,]{.mark}

[& A%D%R,A%X%R,X%R,B%R,INFOGR,A%TYPEXT)]{.mark}

[形参：]{.mark}

[ (NPOIN,NSEGB,GLOSEG,MAXSEG,DA,XA,XINC,RHS,INFOGR,TYPEXT)]{.mark}

首先，需要将各子进程的A矩阵和B向量整合到主进程中，

TELEMAC2D模型中：

streamline.F GLOB_CHAR_COMM() P_MPI_ALLTOALL() MPI_ALLTOALL

[SUBROUTINE P_ALLGATHERV_I]{.mark} （未使用）
