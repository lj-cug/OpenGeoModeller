#include <stdio.h>
#include <stdlib.h>

extern "C" 
void print_c(char *s)
{
  printf("%s \n", s);
}


/*
void print(char *str);
void main(){
char test[] = "Test to print out string in C.";
char *test2;
   test2 = test;
   print_c(test2);
}
*/