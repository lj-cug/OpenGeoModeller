# OpenGL中的纹理操作
## 创建和初始化纹理
OpenGL中的纹理代表图像(image)，可以包装成物体(character)。例如，下面的图像显示了没有纹理和具有纹理的物体。

![](./media/image3.png)

OpenGL中，有2种数据存储类型：Buffers和Textures

Buffers是无类型数据的线性块，可视为通用的内存分配。

Textures是多维数据，比如图像。

![](./media/image4.jpeg)

在OpenGL中，属性(attribute)属于包括：

-   节点位置

-   法向向量

-   U-V坐标

都存储在OpenGL的buffers中。相反，图像数据存储在OpenGL的纹理对象(texture
object)中。

![](./media/image5.jpeg)

为了存储图像数据在一个纹理对象中，必须：

1、创建纹理对象；

2、分配纹理存储空间；

3、绑定纹理对象。

可以使用以下函数创建、绑定和分配纹理存储空间：

// create a new texture object v4.5 opengl

glCreateTextures(\...); // glGenTextures(...) v3.3

// Allocate texture storage

glTexStorage2D(\...);

// Bind it to the GL_Texture_2D target

glBindTexture(\...);

OpenGL在4.5为所有的glGen\*()函数增加了glCreate\*()版本。二者区别如下：glGen\*()只有在bind之后才会生成真正的object，而glCreate\*()在create时立刻生成真正object。

代码实例如下：

//The type used for names in OpenGL is GLuint

GLuint texture;

//创建一个新的2D纹理对象

glCreateTextures(GL_TEXTURE_2D,1,&texture);

//Specify the amount of storage we want to use for the texture

glTextureStorage2D(texture, // Texture object

1, // 1 mimap level

GL_RGBA32F, //32 bit floating point RGBA data

256, 256); //256 x 256 texels

//Bind it to the OpenGL Context using the GL_TEXTURE_2D binding point

glBindTexture(GL_TEXTURE_2D, texture);

在绑定了纹理对象后，就可以向纹理加载数据了。数据通过使用下列函数加载到纹理：

// function to load texture data

glTexSubImage2D(\...)

下列代码展示了如何向纹理加载数据：

// Define some data to upload into the texture

float \*data=new float\[256\*256\*4\];

glTexSubImage2D(texture, //Texture object

0, //level 0

0, 0, //offset 0,0

256, 256, //256 x 256 texels

GL_RGBA, //Four channel data

GL_FLOAT, //Floating point data

data); //Pointer to data

## 纹理目标和类型
纹理目标(target)决定了纹理对象的类型(type)

例如，我们创建的是2D纹理。因此，纹理对象绑定到一个2D纹理目标：GL_TEXTURE_2D。如果是1D纹理，则绑定纹理对象到1D纹理目标：GL_TEXTURE_1D

因为要使用图像，用到很多GL_TEXTURE_2D，这是因为图像是2D表征的数据。

## 在着色器中从纹理读取数据
当纹理对象绑定后，包含数据，着色器可读取使用数据。在着色器中，纹理可声明为统一的采样变量(uniform
Sampler Variables)。

![](./media/image6.jpeg)

例如：

uniform sampler2D myTexture;

采样器维度对应纹理的维度。由于我们的纹理是2D纹理，因此采样器必须为2D，表示2D纹理的采样器类型类型是sampler2D

如果纹理是1D，采样器是sampler1D；如果纹理是3D，则sampler3D

## 纹理坐标
采样器(sampler)表示纹理和采样参数。而纹理坐标表示0.0\~1.0的坐标。需要这2套信息将纹理应用于对象。现在已知了：使用Sampler表示着色器中的纹理。但如何在着色器中表示纹理坐标呢？

物体数据通过OpenGL缓冲发送到GPU。加载这些缓冲，缓冲包含代表物体属性的数据。这些属性可以是节点位置、法向向量或纹理坐标。纹理坐标也称为UV坐标。

一个节点着色器(Vertex Shader)通过称为节点属性(vertex
attribute)接收该信息。节点着色器仅接收节点属性数据。细分(tessellation)、几何和片段着色器不能接收节点属性。如果这些着色器需要该数据，必须从节点着色器传递给他。

![C:\\Users\\Administrator\\Desktop\\OpenGL+Buffer-+UV+coords+(1).jpeg](./media/image7.jpeg)

因此，节点着色器通过节点属性变量接收纹理坐标，然后将坐标传递给片段着色器。当片段着色器中获得纹理坐标后，OpenGL可将纹理应用于对象。

下列代码显示了纹理坐标声明为节点属性。在主函数中，坐标传递给没有修改的片段着色器。

#version 450 core

//uniform variables

uniform mat4 mvMatrix;

uniform mat4 projMatrix;

// Vertex Attributes for position and UV coordinates

layout(location = 0) in vec4 position;

layout(location = 1) in vec2 texCoordinates;

//output to fragment shader

out TextureCoordinates_Out{

vec2 texCoordinates;

}textureCoordinates_out;

void main(){

//calculate the position of each vertex

vec4 position_vertex=mvMatrix\*position;

//Pass the texture coordinate through unmodified

textureCoordinates_out.texCoordinates=texCoordinates;

gl_Position=projMatrix\*position_vertex;

}

## 操控如何读取纹理数据
采样器接收在纹理单元中发现的纹理数据。一个纹理单元包含一个纹理对象和一个采样器对象。上文已经介绍了纹理对象，下面介绍[采样器对象]{.mark}。

纹理坐标范围在0.0\~1.0.
OpenGL允许用户自己决定如何处理在上述范围以外的坐标，称之为[Wrapping
Mode]{.mark}。OpenGL还允许用户决定当纹理上的纹理点(texel)与像素没有1:1比例时如何处理，称之为[Filtering
Mode]{.mark}。采样器对象存储wrapping和filtering参数来控制纹理。

![C:\\Users\\Administrator\\Desktop\\Sampler+Object.jpeg](./media/image8.jpeg)

采样器需要绑定到一个纹理单元的纹理对象和采样器对象。当这套数据完成时，采样器用于实施一个纹理所需要的所有信息。

![C:\\Users\\Administrator\\Desktop\\Sampler+with+texture+unit.jpeg](./media/image9.jpeg)

在特殊情况下，可以创建采样器，将其绑定于一个纹理单元。但大多数情况，你不必创建采样器对象。这是因为一个纹理对象拥有一个默认的采样器对象，可供使用。默认的采样器对象拥有一个默认的wrapping/filtering模式参数。

为了访问默认的采样器对象（存储在纹理对象内部），可以调用：

//accessing the default sampler object in a texture object and setting
the sampling parameters

glTexParameter()

在绑定纹理对象到纹理单元之前，你必须激活纹理单元。通过调用下面的函数来实现，带有需要使用的纹理单元：

//Activate Texture Unit 0

glActiveTexture(GL_TEXTURE0);