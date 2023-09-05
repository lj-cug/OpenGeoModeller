# Computer Graphics

�����ͼ��ѧ�����Լ�һЩ����CPU��GPU������׷�ٵ�ͼ����Ⱦ�����

������ܼ� ./doc/��ѧ���ӻ���ͼ����Ⱦ�ĳ������20220210.docx

## OpenGL

CG������OpenGL��ѧϰ�ĵ����Լ�CUDA-OpenGL������ѧϰ����

## Off-screen Rendering

OSMesa Off-screen Rendering��ParaView�Ļ���EGL��Off-screenͼ����Ⱦ

## Ray-tracing Libraries

### POVRay

The Persistence of Vision Ray Tracer, or POV-Ray������CPU�Ĺ���׷����Ⱦ��

### OSPRay

Intel��˾�з��Ļ���CPU�Ĺ���׷����Ⱦ��

### Nvidia-IndeX��OptiX

Nvidia��˾�з��Ļ���GPU�Ĺ���׷����Ⱦ��

### proland-4.0

����INRIA�з������ڵ����ѧͼ����Ⱦ�Ŀ�

## VTK (Visualization ToolKit)

CFD���õĿ��ӻ�ͼ�ο�

## IceT

��Ⱥͼ����Ⱦ��

�ο���Haeyong Chung et al., A survey of Software frameworks for cluster-based large high-resolution displays  IEEE, 2014, 20(8)

## Blender

��Դ��ͼ����Ⱦ����

## KML

Google Earth��ͼ������

# �����ͼ����Ⱦ(����׷��)

## POV-Ray ����Դ��

POV-Ray��ȫ����[Persistence of Vision
Raytracer]{.mark}����һ��ʹ�ù��߸��ٻ�����άͼ���[����Դ����](https://baike.baidu.com/item/%E5%BC%80%E6%94%BE%E6%BA%90%E4%BB%A3%E7%A0%81/114160)��������

����POV[�ű�����](https://baike.baidu.com/item/%E8%84%9A%E6%9C%AC%E8%AF%AD%E8%A8%80)�����ǻ���DKBTrace��������,
DKBTrace���� David Kirk Buck�� Aaron A. Collins��д�� Amiga�ϵ�.
POV-ray����Ҳ�ܵ���Polyray raytracer ���� Alexander Enzmann
�İ������ܶ�Ư����ͼƬ������POV-ray�������ġ�

����������չʼ��80�����, David Kirk
Buck������һ��Ϊ?[Unix](https://baike.baidu.com/item/Unix/219943)��д��[Amiga](https://baike.baidu.com/item/Amiga/10443049)���߸��������
source code .
��Ȥ���ǣ���������һ��ʱ���������Ӧ���Լ�дһ�����������������ֽ�DKBTrace
�� ������������һ����̳���棬��Ϊ���˻��������Ȥ�� 1987, Aaron
Collins������DKBTraceȻ��ʼ��?[x86](https://baike.baidu.com/item/x86/6150538)��������ֲ����.
����David
Buckһ�����Ϊ������˸���Ĺ������ԡ�ֱ�����������ӵ����У������Ѿ�Ϊ�˼��¹��ܶ�Ӧ����������ʱ��
1989, David
�������Ŀ�����һ������Ա�ŶӺ����Ĺ��̡���ʱ���������Ѿ�û���ʸ���������������ˡ����Կ����˺ܶ��µ����֡�\"STAR\"
(Ϊ��������Ⱦ������������Software Taskforce on Animation and
Rendering) ��һ������Ŀ���, �����������
\"�������ϸ��°汾�Ĺ��߸�������Persistence of Vision Raytracer,\"
��дΪ\"POV-Ray\"

## Intel OSPRay��CPU��

OSPRay��[Intel��˾]{.mark}��[��ѧ���ӻ�����]{.mark}Ŀǰ�Ƚ���CPU����׷�����棨��ܣ������ڵײ��[embree����׷��]{.mark}��ܽ��й����ģ��������ΪOSPRay�������˸������ݵĸ��߼��Ŀ�ܣ�Ŀ�������ڿ�ѧ���ӻ����������Ⱦ���߹���׷����ȾӦ�á�

**ʵʱ����׷�ٷ��ӿ��ӻ�**

OSPRay����������ѧ���ӻ�Ӧ�ÿ�ν�Ƿǳ����㣬��Ϊ�����ܸܺ߼����ˣ��������ģ�Ͷ��Ѿ���װ���ˣ�������Ⱦ��Ҳ�Ѿ���װ���ˣ�������ʱ��ֱ�ӵ��þ��У�ֻ��Ҫ�����������ò���������Ҫд����ĵײ���롣��Ϊ�ǻ���[embree���]{.mark}�����ģ�����OSPRayҲ��ֻ��ҪCPU���������Ⱦ��û��GPUҲ��ȫû���⡣

**���߹���׷��"Embree"**��**������Intel���п�����һϵ�и����ܹ���׷���ں�**���Ż�֧��SSE��AVX�����´�����ָ����ɽ�����Ƭ������Ⱦ���ٶ�Ҳ������������100�������������ṩ��һ����Ƭ����Ⱦ����ʵ����

[Embree]{.mark}ʹ�õ���**���ؿ���(Monte
Carlo)����׷���㷨**�����о���������߶��ǲ���صġ�

Ӣ�ض��Ƴ� oneAPI ��Ⱦ���߰���ӵ�й���׷�ٺ���Ⱦ����.

IT֮�� 8 �� 27 ����Ϣ ����Ӣ�ض��ٷ�����Ϣ���� SIGGRAPH 2020
�����ϣ�Ӣ�ض�������oneAPI
��Ⱦ���߰������²�Ʒ��Ӣ�ض���ʾ����Ⱦ���߰���Ϊͼ������Ⱦ��ҵ�������������ܺͱ���ȡ�

## AMD Radeon ProRender

�Ѿ���Blender��ʹ�á�

## Nvidia IndeX��GPU��

1.  ΢���DirectX
    Raytracing��DXR��API��������׷�ٹ�����ȫ���ɵ���Ϸ�����������õ���ҵ��׼API
    DirectX�У�ʹ����׷�ٳ�Ϊ��դ�������Ĳ��䣬���������Ʒ��DXRרע��ͨ����դ���͹���׷�ټ����Ļ���ͼ������������û�������

2.  NVIDIA��Vulkan����׷����չ������Vulkanͼ�α�׼�Ĺ���׷����չ����Ҳ���ڿ�ƽ̨API��ʵ�ֹ���׷�ٺ͹�դ������������ϵ���һ��;����

3.  NVIDIA��OptiX
    API���ǻ���GPUʵ�ָ����ܹ���׷�ٵ�Ӧ�ó����ܡ���Ϊ���ٹ���׷���㷨�ṩ��һ���򵥡��ݹ������Ĺ��ߡ�OptiX
    SDK�����������໥����ʹ�õ���Ҫ�����������Ⱦ�������Ĺ���׷�������post
    process����������������ʾ�����ء�

NVIDIA IndeX is a [3D volumetric interactive visualization SDK]{.mark}
that allows [scientists and researchers]{.mark} to visualize and
interact with massive data sets, make real-time modifications, and
navigate to the most pertinent parts of the data, all in real-time, to
gather better insights faster. IndeX leverages GPU clusters for
scalable, real-time, visualization and computing of multi-valued
volumetric data together with embedded geometry data.

PARAVIEW PLUGIN FOR WORKSTATIONS AND HPC CLUSTERS

There are two versions of the plug-in. For usage in a workstation, or
single server node, the plug-in is available at no cost. For performance
at scale in a GPU-accelerated multi-node system, the Cluster edition of
the plug-in is available at no cost to academic users and with a license
for commercial users.

## NVIDIA Optix

[Ingo Wald]{.mark}

# ��ѧ���ӻ�������ͼ����Ⱦ�Լ�һЩ�������

����������ѧ���ݵĹ�ģ�������󣬿�ѧ���ӻ������ķ�չѸ�ͣ�������������µĿ�ѧ���ݿ��ӻ��������ͼ����Ⱦ���������룬��һ��ǰ�ؼ���������github���ռ���һЩ���������Ҫ��ʹ��[ParaView�����ͼ����Ⱦ��Դ������Blender��]{.mark}ʵ�ֵ�ͼ�ο��ӻ��⣬��¼�ͽ������£�

## �����ѧ���ݿ��ӻ�

[mobigroup]{.mark}�����˺ܶ�����ѧ���ݣ�ʹ��ParaView���ӻ��ĳ����磺MantaFlow,
����3D�߿�ģ�͡�������������.......

pv_atmos: ����ParaView Python
API���ӻ������ͺ����netcdf��ʽ���ݵĳ���

PVGeo: ��Ҫ�ǵ����������ݵĿ��ӻ����ṹ���ݡ�

PyVista: Python VTK

## Blender��Ⱦ����

### ParaView-Omniverse-Connector

ParaViwe
5.10��ʹ��Connectorͨ��Omniverse�������ܶ�ͼ����Ⱦ�����������Blender��ʵ������������

### Blender��VTK���

������Ϣϵͳ����Ⱦ��Blender GIS (https://github.com/domlysz/BlenderGIS)

## ����

### Paraview-to-POVRay-Water-Render 

����ֵ�������Ȼ��ʹ�ù�׷����������Ⱦ����Ȼˮ������ӣ�

### VTK2Blender

BVKTNodes��ֱ�ӽ�VTK���ṹ�ͷǽṹ�����ݶ�ȡ��Blender������Ⱦ

<https://bvtknodes.readthedocs.io/en/latest/BVTKNodes.html>

### Vulkan

Vulkanһֱ�����������ڶ���������ͼ����Ⱦ����2016�꿪ʼ������Rossant��Vulkan�����ѧ���ݿ��ӻ��о���������Datoviz����������ʼ�׶Ρ�

Cyrille Rossant, Nicolas Rougier. High-Performance Interactive
Scientific Visualization With Datoviz via the Vulkan Low-Level GPU API.
Computing in Science and Engineering, Institute of Electrical and
Electronics Engineers, 2021, 23 (4), pp.85-90.
