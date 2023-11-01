# Hydrology
 
   分布式水文模型及其耦合模型，包括：
   
   1. PIHM - OpenMP并行的三角形非结构网格的分布式水文模型(版本有PIHM2.x, PIHM4.0, SHUD)
   2. ParFLOW - 地表水与地下水模型MODFLOW的耦合模型(C语言开发核心代码, TCL和Python脚本)
   3. GSFLOW - USGS开发的地表水与MODFLOW的耦合模型(pywatersed的前身)
   4. CHM - 加拿大开发的基于非结构网格的冰雪下垫面的分布式水文模型
   5. pywatersed - USGS开发的耦合PRMS水文模型与MODFLOW6地下水模型(Python脚本)
   6. UniFHy - 英国开发的耦合地表水-地下水的水文模型(Python脚本)
   7. tRIBS - 美国开发的基于TIN网格的分布式水文模型(C++语言, MPI并行化)
   8. Raven - 加拿大Waterloo大学研发的集中式和半分布式水文模型(无并行化)
   9. SEIMS - 中国科学院地理所研发的分布式水文模型(C++, MPI + OpenMP并行化)
   
# 总结
   
   1. Python语言开发的水文模型(pywatershed, UniFHy)降低了复杂的水文模型的使用难度，但并行化仍然是未来分布式水文模型的发展趋势(ParFlow)
   2. 非结构化网格能体现出流域的空间各向异性, 但数据结构较结构网格复杂, 目前较少应用于分布式水文模拟, 仅有: PIHM, SHUD, CHM, tRIBS
   3. 分布式水文模型的并行化程度不高, 并行化的水文模型有: CHM, PIHM, SHUD (OpenMP)和tRIBS, SEIMS, ParFLOW (MPI), ParFLOW是并行化模式最多的水文模型(MPI + OpenMP/CUDA)
