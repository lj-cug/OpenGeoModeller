# OpenGL交互式输入设备函数
在OpenGL程序中，交互设备输入有GLUT中的子程序处理，GLUT有从标准输入设备（包括鼠标、键盘、数据板、空间球、按钮盒和拨号盘）接受输入的函数。我们来看一个示例。

代码见demo/code1.cpp

![](./media/image1.png)

在这个示例中，程序对我们的鼠标消息做出响应，在鼠标点击的位置绘制红色的点。这里关键的函数有两个，一个是鼠标消息响应函数：

glutMouseFunc(mousePtPlot);

另一个是回调函数;

void mousePtPlot(GLint button, GLint action, GLint xMouse, GLint yMouse)

该函数有四个参数，分别对应鼠标按钮（GLUT_LEFT_BUTTON、GLUT_MIDDLE_BUTTON、GLUT_RIGHT_BUTTON）、按钮行为（GLUT_DOWN或GLUT_UP）、鼠标响应位置。

下面的示例展示了直线段绘制程序：

demo/code2.cpp

![](./media/image2.png)

可以使用的另一个GLUT鼠标子程序是：

glutMotionFunc(fcnDoSomething);

当鼠标在窗口内移动并且一个或多个鼠标按钮被激活时，这个例程调用下面的函数：

void fcnDoSomthing(GLint xMouse,GLint yMouse)

其中参数是当鼠标被移动并且按钮被按下时，鼠标光标相对于窗口左上角的位置。

类似地，可以使用的另一个GLUT鼠标子程序是：

glutPassiveMotionFunc(fcnDoSomethingElse);

当鼠标在窗口内移动并且没有一个或多个鼠标按钮被激活时，这个例程调用下面的函数：

void fcnDoSomthingElse(GLint xMouse,GLint yMouse)

其中，参数是当鼠标被移动时，鼠标光标相对于窗口左上角的位置。\
GLUT键盘响应函数是：

glutKeyboardFunc(keyFcn);

当键盘某个键被按下时，这个例程调用下面的函数：

void keyFcn(GLubyte key,GLint xMouse,GLint yMouse)

其中，参数是当某个键被按下时，鼠标光标相对于窗口左上角的位置。\
我们看一个简单的示例：

demo/code3.cpp

![](./media/image3.png)

该程序捕获键盘消息，当键盘c键按下时，开启绘点功能。

我们可以使用以下命令指定对于功能键、方向键及其他特殊键的处理函数：

glutSpecialFunc(specialKeyFcn);

回调函数：

void specialKeyFcn(GLint specialKey,GLint xMouse, yMouse);

功能键的符号常量从GLUT_KEY_F1到GLUT_KEY_F12，方向键类似GLUT_KEY_UP，其他特殊件类似GLUT_KEY_PAGE_DOWN。我们来看一段示例代码：

demo/code4.cpp

![](./media/image4.png)

在这个程序中，鼠标消息用于确定方块位置，键盘输入用于缩放方块的大小。每次单击鼠标左键生成一个当前大小的正方形。
