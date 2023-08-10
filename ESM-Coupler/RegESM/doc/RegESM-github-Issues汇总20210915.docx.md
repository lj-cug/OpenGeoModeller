# RegESM-github-Issues汇总

下面将github上RegESM的Issue汇总，期望指导后期开发需要。

## 1、当大气模拟区域大于海洋模拟区域时，超过海洋模拟区域的SST部分由再分析数据补充，在海洋模拟区域边界处由于插值引起梯度的剧烈变化，可能导致不一致的问题。这需要在driver程序中实施光滑算法，使SST逐渐过渡，减小剧烈梯度变化。[rfarneti](https://github.com/rfarneti) commented [on 14 May 2015](https://github.com/uturuncoglu/RegESM/issues/12#issue-76305571)

[已解决。]{.mark}

## 2、driver中的风向旋转仅对Lamber Conformal投影有效，但可能使用不同类型的地图投影：

Africa, Central America and South-East Asia is NORMER

India and South America is ROTMER

此时，需要更新风向旋转代码，支持NORMER和ROTMER。否则，这些计算域会有风向的问题，也会影响海洋流场计算。

风向旋转代码在mod_esmf_atm.F90的尾部。

查看RegCM代码后，发现PreProc/ICBC/mod_uvrot.f90中有。可能使用uvrot4nx子程序来实施ROTMER，但没有发现NORMER的相关风向旋转代码。

[解决：]{.mark}The rotation algorithm now supports ROTMER (Rotated
Mercator) and NORMER (Normal Mercator) but not POLSTR.

## 3、如果2个模型组件使用[相同的变量名交换数据]{.mark}（mask变量在海洋和大气模式中均定义），则为模型出错（错误与内部场有关）。

Error: [（无效形参）]{.mark}

20170301 231238.889 ERROR PET00 OCN-TO-COP:
src/addon/NUOPC/src/NUOPC_Connector.F90: 745 [Invalid argument]{.mark} -
Ambiguous connection status, multiple connections with identical
bondLevel found for: land_sea_mask\
20170301 231238.889 ERROR PET00 regesm:
src/addon/NUOPC/src/NUOPC_Driver.F90: 1601 [Invalid argument]{.mark} -
Phase \'IPDv05p2b\' Initialize for connectorComp 2 -\> 3: OCN-TO-COP did
not return ESMF_SUCCESS\
20170301 231238.889 ERROR PET00 regesm:
src/addon/NUOPC/src/NUOPC_Driver.F90: 837 Invalid argument - Passing
error in return code\
20170301 231238.889 ERROR PET00 regesm:
src/addon/NUOPC/src/NUOPC_Driver.F90: 309 Invalid argument - Passing
error in return code

## 4、关于atm_input3d与PNG图片的存储开关

问题：check grid for input atm_input3d - NOT DEFINED !!!
我需要将atm_input3d相关文件放在什么路径下吗？

回答：No, atm_input3d is the port that can be used under ParaView. So,
this special port basically carries ATM 3d fields. To activate that port
you need to export 3d fields from ATM to COP component. You could see
the example [exfield.tbl]{.mark} files in
the <https://github.com/uturuncoglu/RegESM/blob/master/docs/04_Usage.md>.
Such as ulev:air_wind_u:3d:bilinear:cross:cross:m/s:m/s:1.0:0.0:F is a
definition for 3d fields and this needs to be under atm2cop section.

问题：如果关闭在线可视化时，同时存储PNG图片文件？

回答：如果不想存储PNG文件，可以在co-processing script
(Python)中修改相关代码，关闭输出PNG文件的开关。

问题：不想存储PNG图片文件，只打算做实施渲染。

回答：在co-processing script
(Python)中激活。渲染与[用户使用的可视化管线]{.mark}相关。一些过滤支持多GPU和多处理器，一些不可以，可参考ParaView手册。

## 其他问题

2018年提出的问题

[Failed to compile
mod_esmf_esm.F90](https://github.com/uturuncoglu/RegESM/issues/19)

Okay it makes sense. In mod_esmf_atm_void.F90, the rc argument of
ATM_SetServices is defined as inout. So, could you change it to only out
and try again. It will probably solve the problem.

[How Can I Define \"Enable PerfCheck\" in Namelist.rc]{.mark}

2015年以前的问题：

[Reproducibility problem \... #13]{.mark}

建议 using \"-fp-model precise\" compiler flag in all the model
components

You probably know this, but you can add extra compile flags like
\"-fp-model precise\" by setting the ESMF_F90COMPILEOPTS and
ESMF_CXXCOMPILEOPTS environment variables before building ESMF.

Response from ESMF Team:

We discussed this issue during the ticket meeting today. At first we
were a bit puzzled by the fact that reproducibility of both your model,
and of the ESMF sparse matrix multiplication would depend on the
\"-fp-model precise\" compiler flag. Then Walter pointed out that the
reductions (which definitely happen in the SMM, and maybe in your
models, too) may be vectorized in the SIMD units of the processor,
leading to partial sums. If the processor supports different order of
summing these partial sums, this may introduce reproducibility issues.

I found this from Intel:\
<https://software.intel.com/en-us/articles/consistency-of-floating-point-results-using-the-intel-compiler>\
You may be interested in this.

Bottom line is that auto-vectorization of reductions are turned OFF when
setting the \"-fp-model precise\" flag. At this point I am assuming that
the SMM bit-for-bit reproducibility can only be guaranteed if the
compiler is instructed to NOT vectorize the reduction loops.

The [bit-to-bit reproducibility]{.mark} becomes important in the
scientific studies when user needs to compare the results of the
different configurations of the same model simulation to check the
effects of the different components. The bit-to-bit reproducibility is
still active research area in the earth system science and needs special
attention to the used model components and the working environment (i.e.
operating system, compiler). The issue is mainly related with the
floating-point arithmetic and the representation of the numbers in the
computer (it has a finite resolution to store the numbers). Due to this
reason, numerical results change run to run of the same executable.
Numerical results also changes between different systems. For example,
using same operating system along with different CPU version (Intel®
Xeon® Processor E5540 vs. Intel® Xeon® Processor E3-1275) probably
affect the results. The problem could be caused by different reasons:
the algorithm, alignment of the data in heap and stack (kind of memory),
task or thread scheduler in OpenMP and MPI parallelism. The order of the
numerical operations has also impact on the rounded results (i.e.
reduction operator in MPI code).

The RegESM model is able to generate bit-to-bit reproducible results
when [specific compiler flags are used to compile/install the model
components and also ESMF library.]{.mark} The detailed information about
the the problem and the possible solution can be found i[n the Section
7.4 of User Guide.]{.mark}

[problem in field time stamp information in three component case
(atm+ocn+rtm) #11]{.mark}

[error when one step interpolation is selected #9]{.mark}

The model gives following error when one step interpolation (no
extrapolation support) is selected in exfield.tbl.

forrtl: severe (193): Run-Time Check Failure. The variable
\'mod_esmf_cpl_mp_cpl_computerh\_\$RH2EXIST\' is being used without
being defined\
Image PC Routine Line Source\
libirc.so 00002B0C7A175A1E Unknown Unknown Unknown\
libirc.so 00002B0C7A1744B6 Unknown Unknown Unknown\
regesm.x 000000000138B9F2 Unknown Unknown Unknown\
regesm.x 0000000001325BCB Unknown Unknown Unknown\
regesm.x 000000000049F817 mod_esmf_cpl_mp_c 549 mod_esmf_cpl.F90

解决：It is fixed by adding rh2Exist = .false. to one step interpolation
section.

There is no any control for the allocation of importField and
exportField #8

In util/mod_config.F90, there is no any control for the allocation of
[importField and exportField]{.mark}. This might cause error when the
code is compiled with debug options like following

-O0 -g -check bounds -traceback -mt_mpi

解决：

The util/mod_config.F90 is fixed by adding control to the loop around
lines 550 and 579.

if (allocated(models(j)%exportField)) then\
\...\
\...\
end if\
..\
..\
if (allocated(models(j)%importField)) then\
\...\
\...\
end if
