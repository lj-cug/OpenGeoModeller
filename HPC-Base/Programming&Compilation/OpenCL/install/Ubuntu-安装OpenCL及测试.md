# Ubuntu OS�°�װOpenCL������

## ��װ

��װ clinfo

sudo apt install clinfo

����clinfo���鿴�Կ��Ƿ�֧��OpenCL

��װ��������

apt-get install opencl-headers ocl-icd-opencl-dev


## ����

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

int main() {

/* Host data structures */
cl_platform_id *platforms;

//ÿһ��cl_platform_id �ṹ��ʾһ���������ϵ�OpenCLִ��ƽ̨������ָ������֧��OpenCL��Ӳ������nvidia�Կ���intel CPU���Կ���AMD�Կ���CPU��?

cl_uint num_platforms;

cl_int i, err, platform_index = -1;

/* Extension data */
char* ext_data;
size_t ext_size;
const char icd_ext[] = "cl_khr_icd";

//Ҫʹplatform��������Ҫ�������衣1 ��ҪΪcl_platform_id�ṹ�����ڴ�ռ䡣2 ��Ҫ����clGetPlatformIDs��ʼ����Щ���ݽṹ��һ�㻹��Ҫ����0��ѯ���������ж���platforms

/* Find number of platforms */

//����ֵ���Ϊ-1��˵�����ú���ʧ�ܣ����Ϊ0�����ɹ�
//�ڶ�������ΪNULL����Ҫ��ѯ�������ж��ٸ�platform����ʹ��num_platformsȡ��ʵ��flatform������
//��һ������Ϊ1������������Ҫȡ���1��platform�����Ը�Ϊ������磺INT_MAX�������ֵ�����Ǿ�˵0������ᱨ��ʵ�ʲ��Ժ��񲻻ᱨ�������ǲ���0��ѯ�������ж���platforms

err = clGetPlatformIDs(5, NULL, &num_platforms);

if(err < 0) {
 perror("Couldn't find any platforms.");
 exit(1);
}

printf("I have platforms: %d\n", num_platforms); //���˼��������ʾΪ2����intel��nvidia����ƽ̨

/* Access all installed platforms */
//����1 ����cl_platform_id��������ռ�

platforms = (cl_platform_id*)

malloc(sizeof(cl_platform_id) * num_platforms);

//����2 �ڶ���������ָ��platforms�洢platform

clGetPlatformIDs(num_platforms, platforms, NULL);

/* Find extensions of all platforms */
//��ȡ�����ƽ̨��Ϣ�������Ѿ�ȡ����ƽ̨id�ˣ���ô�Ϳ��Խ�һ����ȡ������ϸ����Ϣ��

//һ��forѭ����ȡ���е������ϵ�platforms��Ϣ

for(i=0; i<num_platforms; i++)
{
/* Find size of extension data */
//Ҳ�Ǻ�ǰ��һ���������õ����͵��ĸ�����Ϊ0��NULL��Ȼ��Ϳ����õ��������ext_size��ȡ������Ϣ�ĳ����ˡ�?

err = clGetPlatformInfo(platforms[i],
CL_PLATFORM_EXTENSIONS, 0, NULL, &ext_size);

if(err < 0)
{
perror("Couldn't read extension data.");
exit(1);
}

printf("The size of extension data is: %d\n", (int)ext_size);//�ҵļ������ʾ224.?

/* Access extension data */
//�����ext_data�൱��һ�����棬�洢�����Ϣ��

ext_data = (char*)malloc(ext_size);

 clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS,? ? ?

 ? ? ext_size, ext_data, NULL);? ? ? ? ? ? ? ?

 printf("Platform %d supports extensions: %s\n", i, ext_data);?

 //��������������̵����֣��������Կ���Ϣ�ǣ�NVIDIA CUDA?

 char *name = (char*)malloc(ext_size);?

 clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,?

 ext_size, name, NULL);

 printf("Platform %d name: %s\n", i, name);

 //�����ǹ�Ӧ����Ϣ�����Կ���Ϣ��NVIDIA Corporation

 char *vendor = (char*)malloc(ext_size);

 clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR,
    ext_size, vendor, NULL);

 printf("Platform %d vendor: %s\n", i, vendor);

 //���֧�ֵ�OpenCL�汾��������ʾ��OpenCL1.1 CUDA 4.2.1
 char *version = (char*)malloc(ext_size);?
 clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION,
  ext_size, version, NULL);

 printf("Platform %d version: %s\n", i, version);

 //���ֻ������ֵ��full profile �� embeded profile
 char *profile = (char*)malloc(ext_size);

 clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE,
   ext_size, profile, NULL);

 printf("Platform %d full profile or embeded profile?: %s\n", i, profile);

 /* Look for ICD extension */?

//���֧��ICD��һ��չ���ܵ�platform�������ʾ��������Intel��Nvidia��֧����һ��չ����

if(strstr(ext_data, icd_ext) != NULL)

platform_index = i;

//std::cout<<"Platform_index = "<<platform_index<<std::endl;

printf("Platform_index is: %d\n", platform_index);
/* Display whether ICD extension is supported */

if(platform_index > -1)
printf("Platform %d supports the %s extension.\n",

platform_index, icd_ext);

//�ͷſռ�
free(ext_data);
free(name);
free(vendor);
free(version);
free(profile);

}

 if(platform_index <= -1)?

 printf("No platforms support the %s extension.\n", icd_ext);?

 /* Deallocate resources */?
 free(platforms);?

 return 0;

}
