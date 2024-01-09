# julia-jupyter-notebook

https://stackoverflow.com/questions/74444101/unable-to-run-julia-1-7-3-kernel-on-jupyter-lab

```
using Pkg
Pkg.add(¡°IJulia¡±)
using IJulia
installkernel("Julia")
```