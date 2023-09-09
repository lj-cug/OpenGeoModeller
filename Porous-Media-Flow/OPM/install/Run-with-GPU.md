# Run with GPU Solver

使用GPU异构并行加速的线性方程组求解器，如cuSPARSE, AmgCL

参考[github-issue-amgcl-support](https://github.com/OPM/opm-simulators/pull/3346)

运行命令：

flow202110 NORNE_ATW2013.DATA --accelerator-mode=amgcl --matrix-add-well-contributions=true --output-dir=out_amgcl

注意：--accelerator-mode的求解器选择

## 编译后端为CUDA的opm-simulator的问题及解决

### 问题
nvcc amgclSolverbackend.cu

编译错误：
impl_pointer xi=x->impl();

使用GNU gcc-9/g++-9, 重新配置AmgCL
	
### 切换CUDA版本

rm -rf /usr/local/cuda

ln -s /usr/local/cuda-9.2 /usr/local/cuda    # 切换到CUDA-9.2
	
## 运行说明

amgclSolverBackend.cpp代码中，在编译时，可以使用不同的amgcl参数(如选择预处理器或求解器)

一些参数通过Flow的形成那传入，例如：

linear-solver-reduction, linear-solver-max-iter, flow-linear-solver-verbosity

AMGCL默认使用内建的后端，运行在CPU上。也可使用CUDA作为后端，如果CMake发现CUDA。
这时必须设置AMGCL_CUDA=1，在amgclSolverBackend.cpp:solve_system().

也可以在solve_system()中选择不同的预处理器或迭代求解器。

针对Norne算例，求解器使用情况：

Generally working: ilu0, ilu1, damped_jacobi, gauss_seidel, bicgstab, gmres, idrs

Generally not working: amg, spai0, chebyshev, dummy, richardson

## Notes

The BdaBridge has use_gpu and use_fpga to determine if it should use a backend or not, amgcl is neither.
To let amgcl use the CUDA backend, set_source_files_properties(opm/simulators/linalg/bda/amgclSolverBackend.cpp PROPERTIES LANGUAGE CUDA) in CMakeLists.txt is used to mark the file as a CUDA source, instead of a C++ source.

It might be possible to rename the file to .cu, but I'm not sure if a non-CUDA compiler can interpret this.

Using ViennaCL without amgcl converged Flow for NORNE, but with 117k linear iterations

Using vexcl(OpenCL) with amgcl converged Flow for NORNE, but with 155k linear iterations.

Using vexcl(CUDA) with amgcl converged Flow for NORNE, but with 154k linear iterations.

ViennaCL and vexcl did not have satisfying convergence/performance so they were left out, but implementations are available.

## Benchmarks

Notation: A (B+C+D+E), F, G , H

A: Total time

B: Assembly time

C: Linear solve time

D: Update time

E: Pre/post step

F: Overall Linearizations

G: Overall Newton Iterations

H: Overall Linear Iterations

CPU: unknown 16-core Haswell, 2.3GHz, GPU: NVIDIA V100:

Dune:                            708 (164+393+ 84+55), 1866, 1501, 22131

amgcl + cuda:                   1090 (166+775+ 83+53), 1868, 1504, 22204

cusparseSolver:                  447 (165+133+ 83+53), 1846, 1485, 21167

openclSolver (graph_coloring):   612 (210+223+108+58), 2371, 1984, 45230

openclSolver (level_scheduling): 637 (170+313+ 85+56), 1824, 1462, 22133

CPU: Intel Xeon E5-2620 v3 @ 2.4GHz, GPU: NVIDIA 2080Ti:

Dune:                            816 (322+413+111+65), 1866, 1501, 22131

amgcl + cuda:                    695 (207+292+116+65), 1866, 1501, 22333

cusparseSolver:                  543 (205+155+106+62), 1845, 1485, 21132

openclSolver (graph_coloring):   725 (267+229+142+71), 2335, 1943, 43688

The value of 'prm.precond.relax.solve.iters' is denoted as N.

CPU: AMD Ryzen 5 2400G, NVIDIA GTX 1050Ti:

Dune:                            414 (120+194+59+34), 1866, 1501, 22131

amgcl + cuda:                    968 (138+720+66+36), 1835, 1474, 22190

amgcl + vexcl, N=2:             1420 (155+1154+76+27), 2042, 1664, 153288 | 329 fallbacks to Dune

CPU: AMD Ryzen 5 2400G, NVIDIA GTX 1050Ti, masters of 2021-8-8 10:00:

amgcl + cuda:                   1028 (136+ 714+67+94), 1860, 1494,  21946 | 0 fallbacks

amgcl + vexcl, N=2:             1363 (155+1024+74+94), 2057, 1678, 154282 | 333 fallbacks

amgcl + vexcl, N=3:             1131 (146+ 808+69+93), 1994, 1619, 128277 |  46 fallbacks

amgcl + vexcl, N=4:              978 (138+ 671+64+90), 1873, 1510,  90303 |   5 fallbacks

amgcl + vexcl, N=5:              964 (136+ 661+63+90), 1850, 1486,  75865 |   0 fallbacks

amgcl + vexcl, N=6:              997 (138+ 688+65+91), 1868, 1503,  69282 |   1 fallbacks

amgcl + vexcl, N=8:             1006 (135+ 703+62+90), 1830, 1466,  57102 |   0 fallbacks

amgcl + vexcl, N=9:             1032 (135+ 732+62+88), 1854, 1491,  54532 |   0 fallbacks

amgcl + vexcl, N=10:            1058 (135+ 758+62+88), 1852, 1491,  51746 |   1 fallbacks

Well plots verified for N=[2,6,8,10]


Detail, amgcl + cuda, NVDIA 2080Ti, accumulated time (s):

solve_system: 225.2

get_result:     0.3

convert_data:   7.4

copy_rhs:       0.4

solve_system here also includes the ilu decomposition inside amgcl

### Example amgcl_options.json

{
    "backend_type": "cuda",
    "precond": {
        "class": "relaxation",
	"type": "ilu0",
        "damping": "0.9",
        "solve": {
          "iters": "2"
        }
    },
    "solver": {
        "type": "bicgstab",
        "maxiter": "200",
        "tol": "1e-2",
        "verbose": "true"
    }
}



