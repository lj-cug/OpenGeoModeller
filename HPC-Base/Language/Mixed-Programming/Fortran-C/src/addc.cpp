#include <stdio.h> 
#define add ADD             // Fortran ����ʹ�ã�ȫ����д
#define arrayadd ARRAYADD
#ifdef __cplusplus  /* wrapper to enable C++ usage */
extern "C" {
#endif
void add(float *, float *, float *);
void arrayadd(int *,float *, float *, float *);
#ifdef __cplusplus  /* wrapper to enable C++ usage */
}
#endif

void add(float *a, float *b, float *c)  // C �����а���ַ��ֵ����Ҫʹ��ָ��
 { 
     *c = *a + *b; 
     printf("%f + %f = %f\n", *a, *b, *c); 
 } 


void arrayadd(int *nmax,float *a, float *b, float *c)
{
  int i;
  for(i=0;i<*nmax;i++)
  {
	  c[i]=a[i]+b[i];
      printf("%f + %f = %f\n", a[i], b[i], c[i]); //
  }
}

 
