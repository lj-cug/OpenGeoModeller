# Windows 11 OS 安装JAVA-1.8

配置环境变量    
JAVA_HOME=C:\Program Files\Java\jdk1.8.0_331
PATH=C:\Program Files\Java\jdk1.8.0_331\bin
JRE_HOME=C:\Program Files\Java\jdk1.8.0_331\jre

问题及解决
Error: Registry key 'Software\JavaSoft\Java Runtime Environment'\CurrentVersion'

where java
显示：
C:\Windows\System32\java.exe
C:\Program Files\Java\jdk1.8.0_331\bin\java.exe


点击 开始 --> 运行... --> 输入 regedit, 回车 --> 打开注册表
找到 HKEY_LOCAL_MACHINE\Software\JavaSoft\Java Runtime Environment\, 就可以查看注册表属性了，我的 CurrentVersion 是 1.8


Solution:
方式一：删除C:\Windows\System32下的java.exe, javaw.exe, javaws.exe

使用 regedit 查看注册表中的 CurrentVersion
使用 where java 查看路径
让注册表中的 CurrentVersion 和 where 命令找到的第一个 java.exe 的版本保持一致！


然后， 测试jtk

gradlew.bat run -P demo=mosaic/PlotFrameDemo.py
gradlew.bat run -P demo=mosaic.PlotFrameDemo
gradlew.bat run -P demo=ogl.HelloDemo

然后， 测试ipf


Linux OS安装Gradle环境下的jtk (Mines Java Tool) 需要：
-------------------------------------------------

(1) 必须使用Java 8 （不要使用最新版本的java）

1.官网下载

https://www.oracle.com/java/technologies/downloads/#java8

2.解压

tar -zxvf jdk-8u111-linux-x64.tar.gz

3.配置(修改 ~/.bashrc)

export JAVA_HOME=/usr/local/java/jdk1.8.0_281

export JRE_HOME=${JAVA_HOME}/jre

export CLASSPATH=.:${JAVA_HOME}/lib/dt.jar:${JAVA_HOME}/lib/tools.jar

export PATH=${JAVA_HOME}/bin:$PATH

4.更新配置

source ~/.bashrc

5.验证

java -version


(2) 更该下载gradle的国内镜像：

jtk/gradle/wrapper/gradle-wrapper.properties

distributionUrl=https\://services.gradle.org/distributions/gradle-6.6.1-bin.zip

修改为：
distributionUrl=https\://repo.huaweicloud.com/gradle/gradle-3.3-all.zip


(3) 项目依赖仓库配置

对于项目依赖仓库配置，如果只是希望对当前gradle工程生效，可以直接在项目根路径的 build.gradle 中进行配置，配置方法如下：

repositories { 
	maven { url "https://maven.aliyun.com/repository/public" }     # 2个国内的镜像，用1个就行了吧？
	maven { url "https://repo.huaweicloud.com/repository/maven" } 
	mavenLocal()    # 本地没有
	mavenCentral() 
}

(4)  export PATH=/home/lijian/Fault-Interpretation/jtk:$PATH

(5) 清理之前构建的项目

gradlew clean allTests    # Windows
./gradlew clean allTests  # Linux
./gradlew clean


(6) 测试jtk

./gradlew run -P demo=mosaic/PlotFrameDemo.py
./gradlew run -P demo=mosaic.PlotFrameDemo
./gradlew run -P demo=ogl.HelloDemo

if all are OK, then:

（7）测试 ipf

  cd ipf

// 构建ipf.jar库

需要修改build.gradle中的Libs，使ipf能找到，可以使用osv/libs下已经下载好的一些jar库。
