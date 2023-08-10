# 1、SCHISM模型中潮汐边界条件施加

每个开边界上的所有节点上定义潮汐边界条件。这些节点的水位变化由所有从bctides.in文件中的调和分析简单叠加而成。潮汐由调和分析分解为若干种组成：M2,
Z0, Q1......

潮汐势能由下列源项公式计算，代入动量方程：

![Tidal-potential-eq.png](./media/image1.png){width="3.754166666666667in"
height="3.2161297025371827in"}

SCHISM模型中，潮汐边界条件和地球潮汐势在bctide.in文件中定义：

8 5.0 ! ntip, tip_dp !! number of tidal potential constituents, cut off
depth (m). The latter is used to save a little computational time as
tidal potential is negligible in shallow water. To include all depths in
calculation, set a negative depth like -100. 

下面有8种潮汐，小于5.0m就不计算潮汐势能，可节约计算时间。\
M2 !!First frequency name \
2 0.242339 1.405189e-04 1.03272 129.18 !!Species type (1 for dirunal,
and 2 for semi-diurnal); constants (Reid 1990); the last 3 numbers are
angular frequency, nodal factor and arguments which should be same as
those for the tidal b.c. (below) \
S2 !! 2nd freq. name\
2 0.113033 1.454441e-04 1.00000 0.00 \
N2\
2 0.046398 1.378797e-04 1.03272 243.40 \
K2\
2 0.030704 1.458423e-04 0.77955 310.55 \
K1\
1 0.141565 7.292117e-05 0.90207 244.87 \
O1\
1 0.100514 6.759775e-05 0.83992 247.06 \
P1\
1 0.046834 7.251056e-05 1.00000 110.04 \
Q1\
1 0.019256 6.495457e-05 0.83992 1.28 

8 nbfr  !!Number of constituents in tidal b.c. \
M2  !!First freq. name 

1.405189e-04 1.03272 129.18  !! freq., nodal factor and arguments;
使用tide_fac.for程序生成

2 ! nope开边界数目\
59 3 0 0 0  !Atlantic 施加潮汐水位边界\
M2 !First freq. (tidal b.c.)\
0.808688 226.095786  ! M2 amplitude and phase at 1st open boundary node\
0.785793 225.374081\
0.731786 224.540114\
\....\
Q1  !last freq. for this boundary\
0.017919 222.956272\
0.017730 222.310333\
\....\
0.012038 198.774149 \
0.011939 198.804927 ! Q1 amplitude and phase at last node

# 2、生成SCHISM模型开边界潮汐的步骤

（1）生成开边界节点上的潮汐幅度和相位，需要hgrid.gr3文件；

（2）如果hgrid.gr3不是经纬度坐标(WGS84)，需要将其投影为经纬度坐标，使用cpp_bp.f程序，记住要将边界信息部分附加到经纬度坐标文件fort.14.ll后面；

（3）运行ecp.f得到ne_pac4.nos8（把经纬度坐标转换为笛卡尔坐标系统）

（4）确认hgrid.gr3的所有网格节点都位于ne_pac4.nos8内部，将hgrid.ll边界信息部分附加到ne_pac4.nos8后面；

（5）运行genbcs.f得到fort.14.sta\-\--仅包含要使用的第一个开边界（潮汐开边界），将所有要施加潮汐边界的节点号整合到一个文件；

（6）运行intel_deg.f获取[ap.dat]{.mark}（幅度和相位），幅度和相位前面是潮汐成分名称；

../intel_deg ne_pac4.nos8 fort.14.sta webtd.tct [ap.dat]{.mark}

上面的webtd.tct 是由webtide预测软件生成的文件。

可使用WebTide的功能：

Import Tide Marker

Edit Parameters

Get Constitutes
