Multi-GPU or GPU Cluster?

[april 12,
2017](https://blogonparallelcomputing.wordpress.com/2017/04/12/multi-gpu-or-gpu-cluster/) by [sahanays](https://blogonparallelcomputing.wordpress.com/author/sahanays/)

Ok, now having known what a GPU is, next question is what is a cluster?
A **computer cluster** consists of a set of loosely or tightly connected
computers that work together so that, in many respects, they can be
viewed as a single system.Computer clusters have each node set to
perform the same task, controlled and scheduled by software.

So, combining the above thoughts, A **GPU cluster** can be thought of a
computer cluster in which each node is equipped with a Graphics
Processing Unit (GPU). By harnessing the computational power of modern
GPUs very fast calculations can be performed with a GPU cluster. A GPU
cluster can be **homogenous** or **heterogenous. **[In case of a
homogenous GPU cluster, every single GPU is of the same hardware class,
make, and model.]{.mark} In case of a heterogeneous GPU cluster,
Hardware from both of the major Independent hardware vendor can be used
(AMD and nVidia). Even if different models of the same GPU are used
(e.g. 8800GT mixed with 8800GTX) the gpu cluster is considered
heterogeneous.

**MULTIple-Graphics Processing Units** is using two or more graphics
cards in the same PC to support faster animation in video games.
[NVIDIA's Scalable Link Interface (SLI)]{.mark} and [ATI's
CrossFire]{.mark} are examples.

For a multi-GPU PC you can easily use CUDA library itself and if you
connect GPUs with a SLI** bridge**, you will see improvements in
performance.

If you want to use a cluster with GPUs, you may use **CUDA-Aware MPI**.
It is combined solution of MPI standard and CUDA library.
