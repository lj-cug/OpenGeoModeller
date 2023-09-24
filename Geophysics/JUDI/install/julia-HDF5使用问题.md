# HDF5.jl使用问题

using HDF5; 出现错误：

/lib/x86_64-linux-gnu/libcurl.so.4: version `CURL_4' not found (required by /home/runner/.julia/artifacts/2829a1f6a9ca59e5b9b53f52fa6519da9c9fd7d3/lib/libhdf5.so)

[github-issue链接](https://github.com/JuliaIO/HDF5.jl/issues/1117)

[HDF5.jl-binary](https://juliaio.github.io/HDF5.jl/stable/#Using-custom-or-system-provided-HDF5-binaries)

## 解决

安装低版本的HDF5

]

remove HDF5

add HDF5@v0.16.14
