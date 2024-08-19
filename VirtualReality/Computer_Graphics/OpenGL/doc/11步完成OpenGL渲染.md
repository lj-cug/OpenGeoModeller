# 11步完成OpenGL渲染
OpenGL API十分庞大，要记住这么多的函数是很困难的。
OpenGL也是一种状态机器，要记住执行很多步骤！
在OpenGL中执行渲染是很容易的，但很容易忘记这些步骤。

下面请记住执行基本的OpenGL渲染的11个步骤：

开始：假设characterData包含渲染物体的数据：

//Vertex data of character

float characterData\[36\]={1.0,0.4,0.9,1.0,\....};

**Step 1**. Generate a Vertex Array Object:

//1. Generate a Vertex Array Object. 
Assume you have a global GLuint myVertexArrayObject declared.

glGenVertexArrays (1,&myVertexArrayObject);

**Step 2**. Bind the Vertex Array Object:

//2. Bind the Vertex Array Object

glBindVertexArray (myVertexArrayObject);

**Step 3**. Generate a Vertex Buffer Object:

//3. Create a vertex buffer object

GLuint myBuffer;

glGenBuffers(1,&myBuffer);

**Step 4**. Bind the Vertex Buffer Object:

//4. Bind the vertex buffer

glBindBuffer(GL_ARRAY_BUFFER,myBuffer);

**Step 5**. Load Data into the buffer:

//5. Load data in the buffer

glBufferData(GL_ARRAY_BUFFER,sizeof(characterData),characterData,GL_STATIC_DRAW);

**Step 6**. Get Location of Attributes in current active shader:

//6. Get the location of the shader attribute called \"position\".
Assume positionLocation is a global GLuint variable

positionLocation=glGetAttribLocation(programObject, \"position\");

**Step 7**. Get Location of Uniform in current active shader:

//7. Get Location of uniform called \"modelViewProjectionMatrix\".
Assume modelViewProjectionUniformLocation is a global GLuint variable.

modelViewProjectionUniformLocation =
glGetUniformLocation(programObject,\"modelViewProjectionMatrix\");

**Step 8**. Enable the attribute location found in the shader:

//8. Enable the attribute location

glEnableVertexAttribArray(positionLocation);

**Step 9**. Link buffer data to shader attributes:

//9. Link the buffer data to the shader attribute locations and inform
OpenGL about the types of data in bound buffers and any memory offsets
needed to access the data

glVertexAttribPointer(positionLocation,3, GL_FLOAT, GL_FALSE, 0, (const
GLvoid \*)0);

**Step 10**. Draw using data in currently bound and enabled buffers:

//1. Bind the VAO

glBindVertexArray (myVertexArrayObject);

//2. Start the rendering process

glDrawArrays(GL_TRIANGLES, 0, 36);

//3. Unbind the VAO

glBindVertexArray (0);

**Step 11**. Delete previously generated buffer:

glDeleteBuffers(1, &myBuffer);