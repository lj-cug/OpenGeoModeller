# DG-SWEM的网上手册

[https://users.oden.utexas.edu/\~michoski/dgswem_doc/index.html#](https://users.oden.utexas.edu/~michoski/dgswem_doc/index.html)

This is documentation for the discontinuous Galerkin shallow water equations model (DG-SWEM).

## 发展史

Clint Dawson研究团队发展的DG法模型，FORTRAN编程。MPI并行化。

从代码看来，DG-SWEM模型在早期（2001），是基于ADCIRC_v41.11(2001年)的框架下开发的，因此后期DG-SWEM的编码风格及文件格式与ADCIRC模型几乎一致。

从2016年，Prapti
Neupane的代码来看，已经脱离了ADCIRC，但仍然采用了很多ADCIRC模型的源码。

在2017年，Brus对DG-SWEM模型做了较大改进，如曲边单元等技术。

Brus的代码与Neupane的代码区别很大了！

## 下载DG-SWEM

Neupane与Brus版本的dgswe不同, github上下载

## 运行DG-SWEM的基本模块

运行DG-SWEM需要FORTRAN编译器和Python及[sympy](http://sympy.org/en/index.html)库

进入dgswem/work/.

make all

./dgswem_serial

或者

mpirun -np 12 ./dgswem

如果需要耦合DG-SWEM与波浪模型[SWAN](http://swanmodel.sourceforge.net/),
可以参考[online forum](https://groups.google.com/forum/#!forum/dgswem).

运行算例，需要风场数据，可以使用get_winds.py脚本下载。运行新的算例时，需要在makefile中关闭get_winds，可以在makefile中删掉winds来实现.

## 输入文件fort.dg

DGSWE的基本输入与[ADCIRC project](http://adcirc.org/)类似，输入文件的信息
参考[adcirc的输入](http://adcirc.org/home/documentation/users-manual-v50/input-file-descriptions/).

输入文件的主要区别见work路径下的fort.dgfile，一个例子：

1    ! DGSWE

0,2  ! padapt(1=on,0=off), pflag(1=smooth,2=shocks)

1,8  ! gflag(0=fixed,1=dioristic), dioristic tolerance (0-100)

1,1,1 ! pl(low p), ph(high p), px(fixed p)

0.00005 ! slimit (epsilon tolerance in pflag=1 p_enrichment)

10      ! plimit (integer timestep counter for p_enrichment)

1,0.5,2  ! k, ks, L for pflag=2 tolerance:log((k\*p)\*\*(-L\*2))-ks

1          ! FLUXTYPE

2,2 		! RK_STAGE,RK_ORDER

1 		! DG_TO_CG (ignore)

0 		! MODAL_IC

0, 86400 ! DGHOT, DGHOTSPOOL

5 1 ! SLOPEFLAG (only 1, 4-10 work; 5 is best/fast!), weight(default 1!)

0, 0.0001000, 0.00001, 1 ! SEDFLAG, porosity, SEVDM, and \# of layers

1 ! Reaction rate when chemistry is on in (litres/(mol\*days))

23556 ! Number of total elements in mesh

0,-1.0,0.0,2.5e-6,0 ! artdif, kappa, s0, uniform_dif, tune_by_hand

(ZE_ROE+bed_ROE)\*\*-1 \*QX_ROE

(ZE_ROE+bed_ROE)\*\*-1 \*QY_ROE

以下是上述参数的简要描述：

-   **DGSWE**: This parameter describes the input type for flow edges,
    or usually river types. For most intents and purposes it is almost
    always set to 1.

-   **padapt**: The first flag determines whether p-adaptivity or
    p-enrichement is turned on, on = 1, 0 = off. The second flag
    dtermines what type of p-enrichement scheme will be used to
    determined where to enrich. 1 = smooth enriches in areas that are
    very smooth, and 2 = shocks enriches in areas where the solutions
    has large gradients.

-   **gflag**: This is another flag having to do with p-enrichement.
    gflag = 0 means fixed form p-enrichment, which means that
    p-enrichement occurs relative to a global tolerance, while gflag = 1
    means that p-enrichment is only performed on some percentage of the
    total number of cells, given by the diorsitic tolerance (which is
    the second number).

-   **pl, ph, px**: These are the polynomial basis degrees for the top
    degree polynomial ph, the bottom degree polynomial pl, and the fixed
    degree polynomial px. To run at third order, for example, ph =
    pl=px=2. To run p-enrichment from first degree to fifth degree,
    starting at initial projection order 4, would
    be ![pl=1,ph=5,px=4](./media/image1.png).

-   **slimit**: is epsilon tolerance when pflag=1 when smooth type
    enrichment is being used.

-   **plimit**: is the counter that determined how many timesteps are
    taken between each p-enrichement/de-enrichment event.

-   **k, ks, L**: These are the tolerances for the type 2 p-enrichement
    scheme, that go into the formula:
    ![\\log{(kp)\^{-2L}}-ks](./media/image2.png).

-   **Fluxtype**: The fluxtype indicated which numerical flux is used.
    Available fluxes are 1 = local Lax Friedrich's, 2 = Roe Flux, 3 =
    Harten-Lax-van Leer Contact flux, 4 = Nonconservative product flux.
    It should be noted that not all fluxtypes are supported by sediment
    transport.

-   **RK_STAGE**: The RK stage is the stage information for stage
    exceeding order Runge Kutta method. Note that this setting is
    related to the type of RK scheme used, see 编译指令

-   **RK_ORDER**: This is the time integration order for the Runge-Kutta
    method.

-   **DG_to_CG**: This flag can be ignored for now.

-   **Modal_IC**: This determines the initial state of the system.
    Modal_IC=0 is a standard cold start configuration. Modal_IC = 1
    reads in modal initial conditions. Modal_IC = 2 reads in hot start
    files. Modal_IC = 3 reads in nodal initial conditions.

-   **DGHOT**: This flag indicates whether hotstart files will be
    produced (1) or not (0), in order to hotstart the code.

-   **DGHOTSPOOL**: This is the spooling number for the hostart, which
    indicates what timestep frequency the hotstart output will be
    written at.

-   **SLOPEFLAG**: The slopeflag determines which slopelimiter will be
    used in the code. Twelve different slopelimiters are supported,
    though the most common setting is slopeflag = 5, which is the
    Bell-Dawson-Shubin (BDS) limiter.

-   **Weight**: The weight is a free parameter used by some
    slopelimiters, that determines specific tolerances. See the
    slopelimiter paper for more detail.

-   **SEDFLAG**: The sediment flag determines whether or not sediment
    will be tranported or not in the domain. If it is, then sedflag = 1
    and DG-SWEM solves the Exner equation. simultaneously for sediment
    transport. If sedflag = 0, then the bathymetry is stationary.

-   **Porosity**: The porosity is a parameter determined by the sediment
    law in the discharge equation. See the Exner equation for more
    details.

-   **SEVDM**: This is the sediment diffusion coefficient value.

-   **Layers**: This determines how many layers of sediment are running.
    Not all functions support more than a single layer of sediment.

-   **Reaction Rate**: This is the rate of the reaction in
    litres/(mol\*day) when chemical reactions are turned on.
    See [*Compiler
    directives*](https://users.oden.utexas.edu/~michoski/dgswem_doc/Compilerflags.html) for
    more details.

-   **Total elements**: This is the total number of element in the mesh.

-   **Artdif,,kappa,s0,uniform_dif,tune_by_hand**: These are the flags
    that determines whether artificial diffison is turned on or not.
    Artdif = 1 is on, Artdif = 0 is off. the kappa and s0 are tolerances
    that determine how sharp the cutoff for the artificial diffusion is.
    The uniform diffusion flag sets all types of diffusion to the same
    value. The tune_by_hand parameter means that each diffusion type
    will be set in the prep routine.

-   **Last two lines:** The last two lines determine the algebraic form
    of the sediment discharge equation. These are tokenized using a
    fortran parser to support any form for the sediment discharge.

## 编译指令

DG-SWEM has been written in a semi-modular way, in order to preserve
performance on when certain features are not active. As a consequence
many compiler directive are used. The directives can be set in
cmplrflg.mk. Here is a breakdown of the compiler directive
options:

-   **DRKSSP**: Indicates that strong stability preserving Runge Kutta
    will be used for time integration, with stage equal to or exceeding
    order.

-   **DRKC**: Indicates Runge-Kutta-Chebyshev methods will be used for
    time integration, with stage equal to or exceeding order.

-   **DWETDR**: Turns the wetting and drying algorithm on.

-   **DOUT_TEC**: Indicates that tecplot output will be produced
    relative to the NSPOOL parameters.

-   **DSLOPEALL**: This means that all slope limiters will be made
    available for use.

-   **DSLOPE5**: This means only the BDS limiter will be made available
    for use.

-   **DARTDIF**: This means artificial diffusion will be made active.

-   **DSED_LAY**: Indicates that layered sediment evolution will be
    computed.

-   **DTRACE**: This means that a tracer field will be active, that is
    passively advected by the velocity field.

-   **DCHEM**: This means that chemistry is turned on.

-   **DP0**: Indicates that piecewise constants are available for use.

-   **DP_AD**: Means that p-enrichement or p-adaptivity is activated.

-   **[DSWAN:]{.mark} Indicates that the nearshore wave model
    [SWAN](http://swanmodel.sourceforge.net/) will be coupled to the
    solution.**
