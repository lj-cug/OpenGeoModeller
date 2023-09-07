# SMS划分网格

1、File-Open-CAD文件

2、左侧边栏中，右击CAD DATA，点击DATA-> MAP

3、中栏点击Select Feature Arc，下拉菜单中Feature Objects点击clean，再点击Build Polygons

4、 中栏点击Select Feature Arc，全选线段右击Redistribute Vertices，在弹出的对话框中设定空间步长

5、 右击左侧边栏中map data里的cad，在菜单中type->models->Generic 2d models

6、 点击中栏的select Feature Polygon，双击图形，弹出对话框可预览

7、 Feature Objects--clean，Feature Objects->Map->2D Mesh

8、 点击下方工具栏中的mesh module，点击上方下拉菜单的element中的linear->quadratic去除网格中点
以上是生成网格，下面进行高程插值

9 、点击下方工具栏的Scatter model，打开高程文件，类型选择最后一个xyz文件

10 、点击菜单栏的Scatter，interpolate to mesh，勾选map z

11、点击下方工具栏contour options弹出窗口中，左侧选择2d mesh，在右方2d mesh的标签栏中勾选contours，在contours标签栏中的contour method中选择colour fill就可以查看插值结果
插值完成 ，下面定义边界

12、回到主界面，在下方工具栏里点击mesh module，中栏点击creat nodstring创立边界

13、站在进口处，面朝水流方向，进口从右向左；接着画出口，也是从右向左；然后画land boundary，按照逆时针的顺序。
画进出口时，先点击第一点，然后按住shift键点击第二点，回车；

画landboundary时，先点击一点，然后按住ctrl键点击第二点，回车；

14、然后是定义边界，首先要在mesh——define model里面定义边界类型，一般：1，river；2，ocean；3，island；
然后选中具体的边界，右键，assign BC,定义边界类型；注意[此处定义边界的顺序要和上面画边界的顺序一致]

15、右击一个边界，renumber

16、保存工程文件和网格文件

17、运行脚本，转换网格文件  

## 参考文献

Todd Jeffrey Wood. Development of a Finite Element Mesh for the Chesapeake and Delaware Bays. Brigham Young University, 2012.4