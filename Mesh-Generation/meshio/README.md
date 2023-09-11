# meshio

不同格式的网格文件之间的转换

## 安装

conda 环境下：

pip install meshio[all]

conda install -c conda-forge meshio

## 可以转换的格式

Abaqus (.inp), ANSYS msh (.msh), AVS-UCD (.avs), CGNS (.cgns), DOLFIN XML (.xml), Exodus (.e, .exo), FLAC3D (.f3grid), H5M (.h5m), Kratos/MDPA (.mdpa), Medit (.mesh, .meshb), MED/Salome (.med), Nastran (bulk data, .bdf, .fem, .nas), Netgen (.vol, .vol.gz), Neuroglancer precomputed format, Gmsh (format versions 2.2, 4.0, and 4.1, .msh), OBJ (.obj), OFF (.off), PERMAS (.post, .post.gz, .dato, .dato.gz), PLY (.ply), STL (.stl), Tecplot .dat, TetGen .node/.ele, SVG (2D output only) (.svg), SU2 (.su2), UGRID (.ugrid), VTK (.vtk), VTU (.vtu), WKT (TIN) (.wkt), XDMF (.xdmf, .xmf).

## 转换操作

meshio convert    input.msh output.vtk   # convert between two formats

meshio info       input.xdmf             # show some info about the mesh

meshio compress   input.vtu              # compress the mesh file

meshio decompress input.vtu              # decompress the mesh file

meshio binary     input.msh              # convert to binary format

meshio ascii      input.msh              # convert to ASCII format

## 举例

meshio convert    input.e output.plt     # 将Exodus格式转换为Tecplot格式

