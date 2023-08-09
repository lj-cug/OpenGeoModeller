#include "GL\glut.h"
#include "math.h"

int width, height;

void draw()
{
    glutWireSphere(1.5, 20, 30);
}

void display()
{
double Near = 1, Far = 1000, fov = 45;
double ratio = width / height;
double radians, DTOR = 0.0174532925;
double ViewWidth2, ViewHeight2;
double left, right, top, bottom;
double eyesep = 2;

radians = DTOR * fov;                                                                            //�ǶȺͻ��ȓQ��
ViewHeight2 = Near * tan(radians);                                                        //Ӌ��ҕҰ�ĸ߶�  
ViewWidth2 = ViewHeight2 * ratio;                                                       //Ӌ��ҕҰ�Č���

glDrawBuffer(GL_BACK_RIGHT);                                                        //��������ᾏ�n�^
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
glDrawBuffer(GL_BACK_LEFT);
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//����������۵Ĳ���
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
glMatrixMode(GL_PROJECTION);                                                        //�Ȍ�ͶӰ���߀ԭ���λ���
glLoadIdentity();


left = -ViewWidth2 - 0.5 * eyesep * 0.3;                                                //Ӌ�����۵�ҕҰ����
right = ViewWidth2 - 0.5 * eyesep * 0.3;
top = ViewHeight2;
bottom = -ViewHeight2;

glFrustum(left, right, bottom, top, Near, Far);                                         //�������۵�ҕҰ����������ͶӰ���׃���ψD���{ɫͶӰƽ��


gluLookAt(0 + eyesep / 2, 0, 5, 0 + eyesep / 2, 0, -5, 0, 1, 0);                //�Q�����۵�λ��


glDrawBuffer(GL_BACK_RIGHT);                                                        //���VOpenGLʹ���ң��ᾏ�n�^
draw();                                                                                         //�L������
/////////////////////////////////////////////////////////////////////////////////

//����������۵Ĳ���
glMatrixMode(GL_PROJECTION);
glLoadIdentity();


left = -ViewWidth2 + 0.5 * eyesep * 0.3;
right = ViewWidth2 + 0.5 * eyesep * 0.3;
top = ViewHeight2;
bottom = -ViewHeight2;

glFrustum(left, right, bottom, top, Near, Far);

glMatrixMode(GL_MODELVIEW);
glLoadIdentity();

glDrawBuffer(GL_BACK_LEFT);
gluLookAt(0 - eyesep / 2, 0, 5, 0 - eyesep / 2, 0, -5, 0, 1, 0);
draw();
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
glutSwapBuffers();                                                                                       //�����Ҿ��n�^���Q�@ʾ
}

void reshape(int w, int h)
{
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
glViewport(0,0,(GLsizei)w,(GLsizei)h);
width = w;
height = h;
glMatrixMode(GL_PROJECTION);
glLoadIdentity();
gluPerspective(45.0, w / h, 1.0, 1000.0);
glMatrixMode(GL_MODELVIEW);
glLoadIdentity();
}

int main(int argc, char* argv[])
{
glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STEREO);    //GLUT_STEREO����ָ�������Sʹ��OpenGL�����wģʽ
glutInitWindowSize(600, 600);
glutInitWindowPosition(100, 100);
glutCreateWindow(argv[0]);
glutDisplayFunc(display);
glutReshapeFunc(reshape);
glutMainLoop();
return 0;
}