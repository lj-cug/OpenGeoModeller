# 两种方式：

（1）将C语言程序vec(),编译为lib文件，然后再FORTRAN程序中，写：

interface
  subroutine vec(r,len) bind(c)
    use,intrinsic :: iso_c_binding
    implicit none
    integer(c_int) :: len
    real(c_double) :: r(0:len)
   end subroutine vec
end interface

这种方法最常用。注意： FORTRAN与C的传递参数给函数的不同方式（传值或传地址）。

（2）在C语言程序中，编写接口API：

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

当FORTRAN调用以上C语言的lib库时，如果没有定义：

#define add ADD             // Fortran 调用使用，全部大写

需要在Intel FORTRAN编译器中设置：

FORTRAN -》 External Procedures -》Upper Case (/names:uppercase)

注意：32位和64位程序之间的调用！必须匹配。