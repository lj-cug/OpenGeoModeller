#include <stdio.h> 
#define add ADD             // Fortran 调用使用，全部大写
#define arrayadd ARRAYADD
#ifdef __cplusplus  /* wrapper to enable C++ usage */
extern "C" {
#endif
void add(float *, float *, float *);
void arrayadd(int *,float *, float *, float *);
#ifdef __cplusplus  /* wrapper to enable C++ usage */
}
#endif

void add(float *a, float *b, float *c)  // C 语言中按地址传值，需要使用指针
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

 
