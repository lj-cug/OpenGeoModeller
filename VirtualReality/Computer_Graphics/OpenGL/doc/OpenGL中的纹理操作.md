# OpenGL�е��������
## �����ͳ�ʼ������
OpenGL�е��������ͼ��(image)�����԰�װ������(character)�����磬�����ͼ����ʾ��û������;�����������塣

![](./media/image3.png)

OpenGL�У���2�����ݴ洢���ͣ�Buffers��Textures

Buffers�����������ݵ����Կ飬����Ϊͨ�õ��ڴ���䡣

Textures�Ƕ�ά���ݣ�����ͼ��

![](./media/image4.jpeg)

��OpenGL�У�����(attribute)���ڰ�����

-   �ڵ�λ��

-   ��������

-   U-V����

���洢��OpenGL��buffers�С��෴��ͼ�����ݴ洢��OpenGL���������(texture
object)�С�

![](./media/image5.jpeg)

Ϊ�˴洢ͼ��������һ����������У����룺

1�������������

2����������洢�ռ䣻

3�����������

����ʹ�����º����������󶨺ͷ�������洢�ռ䣺

// create a new texture object v4.5 opengl

glCreateTextures(\...); // glGenTextures(...) v3.3

// Allocate texture storage

glTexStorage2D(\...);

// Bind it to the GL_Texture_2D target

glBindTexture(\...);

OpenGL��4.5Ϊ���е�glGen\*()����������glCreate\*()�汾�������������£�glGen\*()ֻ����bind֮��Ż�����������object����glCreate\*()��createʱ������������object��

����ʵ�����£�

//The type used for names in OpenGL is GLuint

GLuint texture;

//����һ���µ�2D�������

glCreateTextures(GL_TEXTURE_2D,1,&texture);

//Specify the amount of storage we want to use for the texture

glTextureStorage2D(texture, // Texture object

1, // 1 mimap level

GL_RGBA32F, //32 bit floating point RGBA data

256, 256); //256 x 256 texels

//Bind it to the OpenGL Context using the GL_TEXTURE_2D binding point

glBindTexture(GL_TEXTURE_2D, texture);

�ڰ����������󣬾Ϳ�����������������ˡ�����ͨ��ʹ�����к������ص�����

// function to load texture data

glTexSubImage2D(\...)

���д���չʾ�����������������ݣ�

// Define some data to upload into the texture

float \*data=new float\[256\*256\*4\];

glTexSubImage2D(texture, //Texture object

0, //level 0

0, 0, //offset 0,0

256, 256, //256 x 256 texels

GL_RGBA, //Four channel data

GL_FLOAT, //Floating point data

data); //Pointer to data

## ����Ŀ�������
����Ŀ��(target)������������������(type)

���磬���Ǵ�������2D������ˣ��������󶨵�һ��2D����Ŀ�꣺GL_TEXTURE_2D�������1D��������������1D����Ŀ�꣺GL_TEXTURE_1D

��ΪҪʹ��ͼ���õ��ܶ�GL_TEXTURE_2D��������Ϊͼ����2D���������ݡ�

## ����ɫ���д������ȡ����
���������󶨺󣬰������ݣ���ɫ���ɶ�ȡʹ�����ݡ�����ɫ���У����������Ϊͳһ�Ĳ�������(uniform
Sampler Variables)��

![](./media/image6.jpeg)

���磺

uniform sampler2D myTexture;

������ά�ȶ�Ӧ�����ά�ȡ��������ǵ�������2D������˲���������Ϊ2D����ʾ2D����Ĳ���������������sampler2D

���������1D����������sampler1D�����������3D����sampler3D

## ��������
������(sampler)��ʾ����Ͳ��������������������ʾ0.0\~1.0�����ꡣ��Ҫ��2����Ϣ������Ӧ���ڶ���������֪�ˣ�ʹ��Sampler��ʾ��ɫ���е��������������ɫ���б�ʾ���������أ�

��������ͨ��OpenGL���巢�͵�GPU��������Щ���壬������������������Ե����ݡ���Щ���Կ����ǽڵ�λ�á������������������ꡣ��������Ҳ��ΪUV���ꡣ

һ���ڵ���ɫ��(Vertex Shader)ͨ����Ϊ�ڵ�����(vertex
attribute)���ո���Ϣ���ڵ���ɫ�������սڵ��������ݡ�ϸ��(tessellation)�����κ�Ƭ����ɫ�����ܽ��սڵ����ԡ������Щ��ɫ����Ҫ�����ݣ�����ӽڵ���ɫ�����ݸ�����

![C:\\Users\\Administrator\\Desktop\\OpenGL+Buffer-+UV+coords+(1).jpeg](./media/image7.jpeg)

��ˣ��ڵ���ɫ��ͨ���ڵ����Ա��������������꣬Ȼ�����괫�ݸ�Ƭ����ɫ������Ƭ����ɫ���л�����������OpenGL�ɽ�����Ӧ���ڶ���

���д�����ʾ��������������Ϊ�ڵ����ԡ����������У����괫�ݸ�û���޸ĵ�Ƭ����ɫ����

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

## �ٿ���ζ�ȡ��������
����������������Ԫ�з��ֵ��������ݡ�һ������Ԫ����һ����������һ�����������������Ѿ���������������������[����������]{.mark}��

�������귶Χ��0.0\~1.0.
OpenGL�����û��Լ�������δ�����������Χ��������꣬��֮Ϊ[Wrapping
Mode]{.mark}��OpenGL�������û������������ϵ������(texel)������û��1:1����ʱ��δ�����֮Ϊ[Filtering
Mode]{.mark}������������洢wrapping��filtering��������������

![C:\\Users\\Administrator\\Desktop\\Sampler+Object.jpeg](./media/image8.jpeg)

��������Ҫ�󶨵�һ������Ԫ���������Ͳ��������󡣵������������ʱ������������ʵʩһ����������Ҫ��������Ϣ��

![C:\\Users\\Administrator\\Desktop\\Sampler+with+texture+unit.jpeg](./media/image9.jpeg)

����������£����Դ������������������һ������Ԫ���������������㲻�ش�������������������Ϊһ���������ӵ��һ��Ĭ�ϵĲ��������󣬿ɹ�ʹ�á�Ĭ�ϵĲ���������ӵ��һ��Ĭ�ϵ�wrapping/filteringģʽ������

Ϊ�˷���Ĭ�ϵĲ��������󣨴洢����������ڲ��������Ե��ã�

//accessing the default sampler object in a texture object and setting
the sampling parameters

glTexParameter()

�ڰ������������Ԫ֮ǰ������뼤������Ԫ��ͨ����������ĺ�����ʵ�֣�������Ҫʹ�õ�����Ԫ��

//Activate Texture Unit 0

glActiveTexture(GL_TEXTURE0);