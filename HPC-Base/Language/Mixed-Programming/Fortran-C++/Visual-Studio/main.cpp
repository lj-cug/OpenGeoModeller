#include <stdio.h>
#include <iostream>
#include <windows.h>
using namespace std;

int main()
{
float c;

//��������Լ��
typedef int (__cdecl * FACT)(int n);
typedef void (__cdecl * PYTHAGORAS)(float a, float b, float *c);

//���ض�̬���ļ�
HINSTANCE hLibrary=LoadLibrary(L"forsubs.dll");
if(hLibrary==NULL)
{
   cout<<"can't find the dll file"<<endl;
   return -1;
}

//���Fortran��������FACT�ĵ�ַ
FACT fact=(FACT)GetProcAddress(hLibrary,"FACT");
if(fact==NULL)
{
   cout<<"can't find the function file."<<endl;
   return -2;
}

//���Fortran��������PYTHAGORAS�ĵ�ַ
PYTHAGORAS pythagoras=(PYTHAGORAS)GetProcAddress(hLibrary,"PYTHAGORAS");
if(pythagoras==NULL)
{
   cout<<"can't find the function file."<<endl;
   return -2;
}

//�׳�
printf("Factorial of 7 is: %d\n", fact(7));

// c = sqrt(a^2+b^2)
pythagoras (30, 40, &c);
printf("Hypotenuse if sides 30, 40 is: %f\n", c);

FreeLibrary(hLibrary); //ж�ض�̬���ļ�
return 0;
}