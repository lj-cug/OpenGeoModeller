# 安装SeismicMesh

安装seismicmesh需要[CGAL](https://www.cgal.org/)，安装CGAL执行：

apt install libcgal-dev

pip install -U SeismicMesh

如果你需要从segy/h5格式文件读写速度文件，则安装：

pip install -U SeismicMesh[io]

## 示例

可以从2D/3D地震速度模型，快速以串行和并行，生成有限单元网格


## 检查

SeismicMesh's mesh generator is sensitive to poor geometry definitions and thus you should probably check it prior to complex expensive meshing. We enable all signed distance functions to be visualized via the ``domain.show()`` method where `domain` is an instance of a signed distance function primitive from `SeismicMesh.geometry`. Note: you can increase the number of samples to visualize the signed distance function by increasing the kwarg `samples` to the `show` method, which is by default set to 10000.

## 并行计算

A simplified version of the parallel Delaunay algorithm proposed by [Peterka et. al 2014](https://dl.acm.org/doi/10.1109/SC.2014.86) is implemented inside the DistMesh algorithm, which does not consider sophisticated domain decomposition or load balancing yet. A peak speed-up of approximately 6 times using 11 cores when performing 50 meshing iterations is observed to generate the 33M cell mesh of the EAGE P-wave velocity model. Parallel performance in 2D is better with peak speedups around 8 times using 11 cores. While the parallel performance is not perfect at this stage of development, the capability reduces the generation time of this relatively large example (e.g., 33 M cells) from 91.0 minutes to approximately 15.6 minutes. Results indicate that the simple domain decomposition approach inhibit perfect scalability. The machine used for this experiment was an Intel Xeon Gold 6148 machine clocked at 2.4 GHz with 192 GB of RAM connected together with a 100 Gb/s InfiniBand network.

To use parallelism see the [docs](https://seismicmesh.readthedocs.io/en/par3d/tutorial.html#basics)

**See the paper/paper.md and associated figures for more details.**

## 效率

**How does performance and cell quality compare to Gmsh and CGAL mesh generators?

Here we use SeismicMesh 3.1.4, [pygalmesh](https://github.com/nschloe/pygalmesh) 0.8.2, and [pygmsh](https://github.com/nschloe/pygmsh) 7.0.0 (more details in the benchmarks folder).

Some key findings:

* Mesh generation in 2D and 3D using analytical sizing functions is quickest when using Gmsh but a closer competition for CGAL and SeismicMesh.
* However, using mesh sizing functions defined on gridded interpolants significantly slow down both Gmsh and CGAL. In these cases, SeismicMesh and Gmsh perform similarly both outperforming CGAL's 3D mesh generator in terms of mesh generation time.
* All methods produce 3D triangulations that have a minimum dihedral angle > 10 degrees enabling stable numerical simulation (not shown)
* Head over to the `benchmarks` folder for more detailed information on these experiments.

![Summary of the benchmarks](https://user-images.githubusercontent.com/18619644/99252088-38e20100-27ed-11eb-80b3-c10afac7efbf.png)

* **In the figure for the panels that show cell quality, solid lines indicate the mean and dashed lines indicate the minimum cell quality in the mesh.**

* Note: it's important to point out here that a significant speed-up can be achieved for moderate to large problems using the [parallel capabilities](https://seismicmesh.readthedocs.io/en/master/tutorial.html#basics) provided in SeismicMesh.


**For an additional comparison of *SeismicMesh* against several other popular mesh generators head over to [meshgen-comparison](https://github.com/nschloe/meshgen-comparison).
