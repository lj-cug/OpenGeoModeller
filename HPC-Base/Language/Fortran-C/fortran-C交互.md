# Fortran��C֮����໥����

https://zhuanlan.zhihu.com/p/613175338?utm_id=0

## Fortran-call-C

use ios_c_binding

Fortran 2003����������ģ��ios_c_binding�������ṩ��һЩc������صĳ����ͺ�����

```
program test
    use iso_c_binding
    implicit none
    integer(kind=c_size_t) :: n
    real(kind=c_long_double) :: x
    logical(kind=c_bool) :: b
    character(kind=c_char) :: c
    print *, huge(n)
    print *, huge(x)
    b = .true.
    print *, b
    c = c_horizontal_tab ! �ȼ��� c ���� '\t'
    print '(A)', c//"Hello world!"
end program test
```

ģ�鶨���˺�c���Եȼ۵�һϵ�л������ͳ������������������Լ�c_int_fast32_t�ȡ�����һ����Fortran��c����֮������ݽ������Ӿ���³���ԡ�

֮ǰ�����������鼼��ʱ�����Ǽ�Ҫע��Fortran��Ĭ��integer��real�ĳ��ȣ���Ҫע��c����Ĭ�ϵ�int���ȣ�����Fortran���ο�ָ�룬һ�����ݳ��ȶԲ��ϣ�����������Եġ�����iso_c_binding�������Щ�������Ͳ��õ�����Щ�����ˡ�

һ���ȽϿӵ��������ǣ�Fortran����û���޷��������ģ�����ios_c_binding��Ҳû���κ�unsigned�汾������������c_size_tʵ����Ҳ���з��ŵġ�

�������ͳ���֮�⣬��������л�չʾ���ַ�����c_horizontal_tab��Fortran�ǲ�֧��ת���ַ��ģ���Щ������������c���Ի��ഫ���ַ���ʱ�����á�

����������ο�Fortran Wiki.

bind�ؼ���

�ڼ�����ios_c_bindingģ��֮ǰ���ȿ�������������õĻ����﷨��

## ��ʼFortran����c

��������Fortran�ļ���

```
program test
    use iso_c_binding, only : c_int
    implicit none
	
    interface
        function f_add1(x) result(ans) bind(c, name="add1")
            import :: c_int
            integer(kind=c_int), intent(in), value :: x
            integer(kind=c_int) :: ans
        end function
    end interface
	
    integer :: i, a
    a = 0
    do i = 1,5
        a = f_add1(a)
    end do
    print *, a
end program test
```

�����õĺ�����c�ļ�

```
#include <stdio.h>

int add1(int x)
{
    printf("called add1\n");
    return x + 1;
}
```

�ȿ�c�ļ���������һ������add1������������int(int)���ٿ�Fortran�ļ���������Ҫдһ��interface����������������ڡ�Ȼ��ʹ��bind�ؼ�����ָ�������������ԡ�

name = "add1"��ָ������������֮������֣��ǿ���ʡ�Եģ�ʡ����ô�ͼٶ����������־��Ǻ�������������c++��extern "C"������Ҳ��������д

function add1(x) result(ans) bind(c)

�����˷��ŵ������������⡣

����֮�⻹��Ҫע������һ����Ҫ�Ĺؼ���value������ָ����x����������Ϊvalue����˼�ǰ���ֵ���ݲ�������ΪFortranĬ�ϵĴ��η�ʽ�����ã�����c���Խ����Ǿͱ���Ϊ��ָ�롣������ǲ���value���ԣ���ôc���Եĺ�����������д

int add1(int *x) {return *x + 1;}

��Ȼ�������б�Ҫ��ָ�룬������ֵ�������Ǹ���ˬ�ġ�

Ŀǰ��֧��bind(c)��c���ԡ�Ҳ����˵�������Ժ��п��ܻ���bind(cpp), bind(rust)��

## C-call-Fortran

������Ϊc����

```
#include <stdio.h>

extern int add(int, int);

int main(int argc, char const *argv[])
{
    printf("%d\n", add(1,2));
    return 0;
}
```

�����õĺ�����Fortran�ṩ

```
module what_ever_module
    use iso_c_binding
    implicit none
contains
    integer(c_int) function add(x, y) result(ans) bind(c)
        integer(c_int), intent(in), value :: x, y
        ans = x + y
    end function
end module what_ever_module
```

�����ǰһ�����ӣ�������Ӿͷǳ��򵥵ġ���Ҫע����ǣ��κ�ģ���еĺ�����bind(c)֮�󣬱�����žͲ��ں���ģ�����Ϣ�ˣ�����add������д���κ�ģ�鶼��һ���ġ�Ҳ��ˣ�bind(c)�ĺ�����������������ʹ�����ڲ�ͬ��module���档

�Ͼ���bind(c)��c������û�к������ص����ԣ�û���������顣

### ios_c_binding�ļ�������

ios_c_binding�г��˳���֮�⣬�������˼����������Լ�c����ָ������type(c_ptr)������ָ������type(c_funptr)����ϧ�������Ƕ��������־����ָ�����ͣ�ֻ��˵�ɿ�һ�ðɡ�

c_loc��c_funloc������fortran��ָ���뺯��ָ��ת��Ϊtype(c_ptr)��type(c_funcptr)��c_f_pointer��c_f_procpointer��c���Ե�ָ�븳ֵ��Fortran��ԭ��ָ�롣

�����÷��뿴Fortran Wiki��ios_c_bindingģ��.

�������һ���Ƚϸ��ӵ�������˵����غ������÷���

c���Բ�������

```
#include <math.h>
#include <string.h>
#include <stdio.h>

float circle(float x) {return 4.0 - x * x;}
float one(float x) {return 1.0;}

typedef float (*float_func)(float);

float_func choose_func(const char *name)
{
    if(strcmp(name, "sin") == 0)
        return sinf;
    if(strcmp(name, "cos") == 0)
        return cosf;
    if(strcmp(name, "circle") == 0)
        return circle;
    return one;
}
```

Fortran��������

```
module integral_m
    use iso_c_binding
    implicit none
    interface
        real function real_func(x)
            real, intent(in), value :: x
        end function
    end interface
contains
    impure real function real_integral(f, a, b, N) result(ans)
        procedure(real_func), pointer, intent(in) :: f
        real, intent(in) :: a, b
        integer, optional :: N
        integer :: Npoints, i
        real :: x, h
        Npoints = 10000
        if(present(N)) Npoints = N
        h = (b - a) / Npoints
        ans = 0.0
        do i = 1, Npoints
            x = a + (i - 0.5) * h
            ans = ans + f(x)
        end do
        ans = ans * h
    end function
end module integral_m

program test
    use iso_c_binding
    use integral_m
    implicit none
    interface
        function choose_func(c_name) bind(c)
            import :: c_ptr, c_funptr
            ! const char*������ `ios_c_binding` ֻ��`c_ptr`����
            type(c_ptr), value, intent(in) :: c_name
            type(c_funptr) :: choose_func
        end function

        real(c_float) function float_func(x)
            import :: c_float
            real(c_float), intent(in), value :: x
        end function
    end interface
    character(len=:), allocatable, target :: name
    procedure(real_func), pointer :: f
    type(c_funptr) :: cf
    ! ʹ�� Fortran ԭ���ĺ���ָ��
    f => circle
    print*, real_integral(f, 0., 2.0)

    name = "sin"
    cf = choose_func(c_loc(name))
    ! c ����ָ�븳ֵ�� Fortran
    call c_f_procpointer(cf, f)
    print*, real_integral(f, 0., 3.14)

    name = "cos"
    cf = choose_func(c_loc(name))
    f => wapper
    print *, real_integral(f, 0., 1.57)

contains
    real function circle(x) result(ans)
        real, intent(in), value :: x
        ans = sqrt(4.0 - x * x)
    end function

    ! ��ȷ�� c �� `float` �� Fortran �� `real` �Ƿ�һ��
    ! �����д�������һ��ת��
    real function wapper(x) result(ans)
        real, intent(in), value :: x
        real(c_float) :: temp
        procedure(float_func), pointer :: ff
        call c_f_procpointer(cf, ff)
        temp = real(x, kind=c_float)
        temp = ff(temp)
        ans = real(temp)
    end function
end program test
```