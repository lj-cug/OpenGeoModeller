# Serafin格式说明

Serafin格式是EDF开发的一种大数据存储格式，类似于NetCDF格式。

Serafin格式文件包括一组网格数据（节点和三角形单元，目前不支持四边形网格）和多组时刻的数据。可用于存储2D和3D的telemac模型[计算结果文件和地形文件]{.mark}。

二进制格式的Serafin文件内容如下（为方便理解，还是用英语）：

-   A record containing the title of the study (72 characters) and a 8
    characters string indicating the type of format (SERAFIN or
    SERAFIND)

-   A record containing the two integers NBV(1) and NBV(2) (number of
    linear and quadratic variables, NBV(2) with the value of 0 for
    Telemac, as quadratic values are not saved so far),

-   NBV(1) records containing the names and units of each variable (over
    32 characters),

-   A record containing the integers table IPARAM (10 integers, of which
    only the 6 are currently being used),

    -   if IPARAM (3) ≠ 0: the value corresponds to the x-coordinate of
        the origin of the mesh,

    -   if IPARAM (4) ≠ 0: the value corresponds to the y-coordinate of
        the origin of the mesh,

    -   if IPARAM (7) ≠ 0: the value corresponds to the number of planes
        on the vertical (3D computation),

    -   if IPARAM (8) ≠ 0: the value corresponds to the number of
        boundary points (in parallel),

    -   if IPARAM (9) ≠ 0: the value corresponds to the number of
        interface points (in parallel),

    -   if IPARAM (8) or IPARAM(9) ≠ 0: the array IPOBO below is
        replaced by the array KNOLG (total initial number of points).
        All the other numbers are local to the sub-domain, including
        IKLE.

    -   if IPARAM (10) = 1: a record containing the computation starting
        date,

-   A record containing the integers NELEM,NPOIN,NDP,1 (number of
    elements, number of points, number of points per element and the
    value 1),

-   A record containing table IKLE (integer array of dimension
    (NDP,NELEM) which is the connectivity table. N.B.: in TELEMAC-2D,
    the dimensions of this array are (NELEM,NDP)),

-   A record containing table IPOBO (integer array of dimension NPOIN);
    the value of one element is 0 for an internal point, and gives the
    numbering of boundary points for the others,

-   A record containing table X (real array of dimension NPOIN
    containing the abscissae of the points),

-   A record containing table Y (real array of dimension NPOIN
    containing the ordinates of the points),

接着，再保存下一时刻的数值，首先保存时间值

-   A record containing time T (real),

-   NBV(1)+NBV(2) records containing the results tables for each
    variable at time T.

# Serafin格式文件读取及可视化

可读取Serafin格式文件的软件很多，包括：

-   Blue Kenue

-   Tecplot

-   MATLAB

-   PyTelTools

-   QGIS_UHM_SerafinReader_v2.0

-   pputils

有一个Fortran的程序，将Serafin二进制格式转换为ASCII格式，方便于理解Serafin格式内容。
