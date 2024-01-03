# Ken Kousen. 巧用Gradle构建Android应用. 电子工业出版社. 2017

## 1 在命令行中执行Gradle构建

使用提供的Grdle wrapper （不需要安装Gradle）或者安装Gradle直接运行。
Grdle wrapper引用gradlew或gradlew.bat脚本，w表示wrapper。wrapper使用了应用程序根目录下的gradle/wrapper下的gradlewrapper.jar和gradle.wrapper.properties文件来启动进程。
distributeURL 显示wrapper会下载并安装Gradle


## 2 执行任务： gradle tasks --all  显示所有的构建任务及每个任务的依赖。

## 3 添加Java库的依赖

在Build.gradle文件的dependencies块添加依赖的组、名字和版本。
默认情况下，应用程序有2个build.gradle文件：一个在顶层目录，另一个用于应用程序本身。后者通常保存在app子目录下。
```
dependencies {
	compile fileTree()
	testCompile ''
	compile ''		
}
```

添加一系列的文件到配种，但又希望将它们添加到一个库中，可以再dependencies中使用files或者fileTree语法：
```
dependencies {
	compile files('libs/a.jar','libs/b.jar')
	compile fileTree(dir: 'libs', include: '*.jar')
}
```

## 4 配置仓库

在Gradle构建文件中配置repositories块
定义repository
repositories块告诉Gradle去哪儿找依赖，使用jecnter()或者mavenCentral()，分别代表默认的Bintray JCenter仓库和公共的Maven中央仓库

默认的JCenter仓库
```
repositories {
 jcenter()   // 这里应用位于https://jcenter.bintray.com的JCenter仓库。
}
```
有2种Maven仓库的快捷方式, 
mavenCentral语法应用位于http://repo1.maven.org/maven2中的Maven2仓库。 mavenLocal()语法引用本地的MAven缓存。
```
repositories {
	mavenLocal()
	mavenCentral()
}
```
从URL添加Maven仓库
```
repositories {
	maven {
		url 'http://repo.spring.io/milestone'
	}
}
```
## 5 升级到新版本的Gradle

(1)添加一个wrapper任务到你的build.gradle文件并生成新的wrapper脚本

./gradlew tasks

嵌入wrapper任务：

./gradlew wrapper --gradle-version 2.12

或者

在顶层build.gradle文件显式指定Gradle wrapper任务
```
task wrapper(type: Wrapper) {
	gradleVersion = 2.12
}
```
修改后，运行 ./gradlew wrapper任务就会生成一些新的wrapper文件。

(2)直接修改gradle-wrapper.properties文件中的distributionUrl的值。
