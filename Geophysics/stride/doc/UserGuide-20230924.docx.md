# StrideDownload

## 文献

Carlos Cueto, et al. 2022. Stride: A flexible software platform for
high-performance ultrasound computed tomography. Computer Methods and
Programs in Biomedicine 221:106855

Jump right in using a Jupyter notebook directly in your browser,
using [binder](https://mybinder.org/v2/gh/trustimaging/stride/HEAD).

The recommended way to install Stride is through Anaconda's package
manager (version \>=4.9), which can be downloaded in:

A Python version above 3.8 is recommended to run Stride.

To install Stride, follow these steps:

git clone https://github.com/trustimaging/stride.git

cd stride

conda env create -f environment.yml

conda activate stride

pip install -e .

You can also start using Stride through Docker:

git clone https://github.com/trustimaging/stride.git

cd stride

docker-compose up stride

which will start a Jupyter server within the Docker container and
display a URL on your terminal that looks something
like *https://127.0.0.1:8888/?token=XXX*. To access the server,
copy-paste the URL shown on the terminal into your browser to start a
new Jupyter session.

## Additional packages

To access the 3D visualization capabilities, we also recommend
installing MayaVi:

conda install -c conda-forge mayavi

and installing Jupyter notebook is recommended to access all the
examples:

conda install -c conda-forge notebook

## GPU support (安装[NVIDIA HPC SDK](https://developer.nvidia.com/nvidia-hpc-sdk-downloads))

The Devito library uses OpenACC to generate GPU code. The recommended
way to access the necessary compilers is to install the [NVIDIA HPC
SDK](https://developer.nvidia.com/nvidia-hpc-sdk-downloads).

wget
https://developer.download.nvidia.com/hpc-sdk/22.11/nvhpc_2022_2211_Linux_x86_64_cuda_multi.tar.gz

tar xpzf nvhpc_2022_2211_Linux_x86_64_cuda_multi.tar.gz

cd nvhpc_2022_2211_Linux_x86_64_cuda_multi

sudo ./install

During the installation, select the *single system install* option.

Once the installation is done, you can add the following lines to
your *\~/.bashrc*:

export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/compilers/bin/:\$PATH

export
LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/compilers/lib/:\$LD_LIBRARY_PATH

export
PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/comm_libs/mpi/bin/:\$PATH

export
LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/comm_libs/mpi/lib/:\$LD_LIBRARY_PATH

# Tutorials

<https://github.com/trustimaging/stride/tree/master/stride_examples/tutorials>

# Examples

<https://github.com/trustimaging/stride/tree/master/stride_examples/examples>

# Stride API Reference

-   [[Utility
    > Functions]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/utility_functions.html)

-   [[Core]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/core.html)

-   [[Problem]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/problem/problem_index.html)

    -   [[Domain]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/problem/domain.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Base]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/problem/base.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Data]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/problem/data.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Problem]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/problem/problem.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Medium]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/problem/medium.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Transducers]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/problem/transducers.html)

-   [[Transducer
    > types]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/problem/transducers.html#transducer-types)

    -   [[Geometry]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/problem/geometry.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Acquisitions]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/problem/acquisitions.html)

-   [[Physics]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/physics/physics_index.html)

    -   [[Problem
        > type]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/physics/problem_type.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Isotropic
        > Acoustic]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/physics/iso_acoustic.html)

-   [[Devito]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/physics/iso_acoustic.html#devito)

    -   [[Isotropic
        > Elastic]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/physics/iso_elastic.html)

-   [[Devito]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/physics/iso_elastic.html#devito)

    -   [[Contrast Agents -
        > Marmottant]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/physics/marmottant.html)

-   [[Devito]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/physics/marmottant.html#devito)

    -   [[Common]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/physics/common.html)

-   [[Devito]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/physics/common.html#devito)

    -   [[Boundaries]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/physics/boundaries.html)

-   [[Devito]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/physics/boundaries.html#devito)

-   [[Base]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/physics/boundaries.html#base)

-   [[Optimisation]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/optimisation/optimisation_index.html)

    -   [[Optimisation
        > Loop]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/optimisation/optimisation_loop.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Loss
        > Functions]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/optimisation/loss.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Optimisers]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/optimisation/optimisers.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Pipelines]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/optimisation/pipelines.html)

-   [[Default
    > pipelines]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/optimisation/pipelines.html#module-stride.optimisation.pipelines.default_pipelines)

-   [[Steps]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/optimisation/pipelines.html#module-stride.optimisation.pipelines.steps.filter_traces)

-   [[Plotting]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/plotting/plotting_index.html)

    -   [[Field
        > plotting]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/plotting/fields.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Point
        > plotting]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/plotting/points.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Trace
        > plotting]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/plotting/traces.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Show]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/plotting/show.html)

-   [[Utils]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/utils/utils_index.html)

    -   [[Filters]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/utils/filters.html)

-   [[Butterworth]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/utils/filters.html#butterworth)

-   [[FIR]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/utils/filters.html#fir)

-   [[FFT]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/utils/fft.html)

    -   [[Wavelets]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/utils/wavelets.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Predefined
        > geometries]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/utils/geometries.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Fetch]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/utils/fetch.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Adding data
        > noise]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/utils/noise.html)

    -   [[Extra
        > operators]{.underline}](https://stridecodes.readthedocs.io/en/latest/stride/api/utils/operators.html)

# Mosaic API Reference

-   [[Running
    > mosaic]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/run.html)

-   [[Runtime]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/runtime/runtime_index.html)

    -   [[Runtime]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/runtime/runtime.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Head]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/runtime/head.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Monitor]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/runtime/monitor.html)

-   [[Strategies]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/runtime/monitor.html#strategies)

    -   [[Node]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/runtime/node.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Worker]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/runtime/worker.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Utils]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/runtime/utils.html)

-   [[Comms]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/comms/comms_index.html)

    -   [[Comms]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/comms/comms.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Serialisation]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/comms/serialisation.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Compression]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/comms/compression.html)

-   [[Core]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/core/core_index.html)

    -   [[Tessera]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/core/tessera.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Task]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/core/task.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Base]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/core/base.html)

-   [[Profiling]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/profile/profile_index.html)

-   [[Types]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/types/types_index.html)

    -   [[Immutable]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/types/immutable.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Struct]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/types/struct.html)

-   [[File
    > manipulation]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/file_manipulation/file_manipulation_index.html)

    -   [[HDF5]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/file_manipulation/h5.html)

-   [[Utils]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/utils/utils_index.html)

    -   [[Event
        > loop]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/utils/event_loop.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Subprocess]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/utils/subprocess.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Logger]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/utils/logger.html)

    ```{=html}
    <!-- -->
    ```
    -   [[Change
        > case]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/utils/change_case.html)

    -   [[Utils]{.underline}](https://stridecodes.readthedocs.io/en/latest/mosaic/api/utils/utils.html)
