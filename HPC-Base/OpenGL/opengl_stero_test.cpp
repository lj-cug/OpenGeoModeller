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

radians = DTOR * fov;                                                                            //角度和弧度換算
ViewHeight2 = Near * tan(radians);                                                        //計算視野的高度  
ViewWidth2 = ViewHeight2 * ratio;                                                       //計算視野的寬度

glDrawBuffer(GL_BACK_RIGHT);                                                        //清除左右後緩衝區
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
glDrawBuffer(GL_BACK_LEFT);
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//以下為控制右眼的部分
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
glMatrixMode(GL_PROJECTION);                                                        //先將投影矩陣還原為單位矩陣
glLoadIdentity();


left = -ViewWidth2 - 0.5 * eyesep * 0.3;                                                //計算右眼的視野範圍
right = ViewWidth2 - 0.5 * eyesep * 0.3;
top = ViewHeight2;
bottom = -ViewHeight2;

glFrustum(left, right, bottom, top, Near, Far);                                         //依照右眼的視野範圍，控制投影矩陣變成上圖的藍色投影平面


gluLookAt(0 + eyesep / 2, 0, 5, 0 + eyesep / 2, 0, -5, 0, 1, 0);                //決定右眼的位置


glDrawBuffer(GL_BACK_RIGHT);                                                        //告訴OpenGL使用右，後緩衝區
draw();                                                                                         //繪出場景
/////////////////////////////////////////////////////////////////////////////////

//以下為控制左眼的部分
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
glutSwapBuffers();                                                                                       //將左右緩衝區交換顯示
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
glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STEREO);    //GLUT_STEREO參數指的是允許使用OpenGL的立體模式
glutInitWindowSize(600, 600);
glutInitWindowPosition(100, 100);
glutCreateWindow(argv[0]);
glutDisplayFunc(display);
glutReshapeFunc(reshape);
glutMainLoop();
return 0;
}