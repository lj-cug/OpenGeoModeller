#include <stdio.h>
#include <stdlib.h>

extern "C" void vec(double *r,int len )
{
  int i;
  double *r2;

  printf("This is in C function vec...\n");
  
  printf("%d\n", len);


  for(i=0; i<len; i++)
    {
       printf("%f\n", r[i]);
    }

  printf("Print put a new vector...\n");
  r2 = (double *)malloc(sizeof(double)*(len));

  for(i=0; i<len; i++)
    {
       r2[i] = r[i];
       printf("%f\n", r2[i]);
    }
}

