# Ken Kousen. ����Gradle����AndroidӦ��. ���ӹ�ҵ������. 2017

## 1 ����������ִ��Gradle����

ʹ���ṩ��Grdle wrapper ������Ҫ��װGradle�����߰�װGradleֱ�����С�
Grdle wrapper����gradlew��gradlew.bat�ű���w��ʾwrapper��wrapperʹ����Ӧ�ó����Ŀ¼�µ�gradle/wrapper�µ�gradlewrapper.jar��gradle.wrapper.properties�ļ����������̡�
distributeURL ��ʾwrapper�����ز���װGradle


## 2 ִ������ gradle tasks --all  ��ʾ���еĹ�������ÿ�������������

## 3 ���Java�������

��Build.gradle�ļ���dependencies������������顢���ֺͰ汾��
Ĭ������£�Ӧ�ó�����2��build.gradle�ļ���һ���ڶ���Ŀ¼����һ������Ӧ�ó���������ͨ��������app��Ŀ¼�¡�
```
dependencies {
	compile fileTree()
	testCompile ''
	compile ''		
}
```

���һϵ�е��ļ������֣�����ϣ����������ӵ�һ�����У�������dependencies��ʹ��files����fileTree�﷨��
```
dependencies {
	compile files('libs/a.jar','libs/b.jar')
	compile fileTree(dir: 'libs', include: '*.jar')
}
```

## 4 ���òֿ�

��Gradle�����ļ�������repositories��
����repository
repositories�����Gradleȥ�Ķ���������ʹ��jecnter()����mavenCentral()���ֱ����Ĭ�ϵ�Bintray JCenter�ֿ�͹�����Maven����ֿ�

Ĭ�ϵ�JCenter�ֿ�
```
repositories {
 jcenter()   // ����Ӧ��λ��https://jcenter.bintray.com��JCenter�ֿ⡣
}
```
��2��Maven�ֿ�Ŀ�ݷ�ʽ, 
mavenCentral�﷨Ӧ��λ��http://repo1.maven.org/maven2�е�Maven2�ֿ⡣ mavenLocal()�﷨���ñ��ص�MAven���档
```
repositories {
	mavenLocal()
	mavenCentral()
}
```
��URL���Maven�ֿ�
```
repositories {
	maven {
		url 'http://repo.spring.io/milestone'
	}
}
```
## 5 �������°汾��Gradle

(1)���һ��wrapper�������build.gradle�ļ��������µ�wrapper�ű�

./gradlew tasks

Ƕ��wrapper����

./gradlew wrapper --gradle-version 2.12

����

�ڶ���build.gradle�ļ���ʽָ��Gradle wrapper����
```
task wrapper(type: Wrapper) {
	gradleVersion = 2.12
}
```
�޸ĺ����� ./gradlew wrapper����ͻ�����һЩ�µ�wrapper�ļ���

(2)ֱ���޸�gradle-wrapper.properties�ļ��е�distributionUrl��ֵ��
