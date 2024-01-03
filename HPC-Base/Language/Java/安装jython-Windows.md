# Installer Jar

[下载Jython-2.7.2](https://www.jython.org/download)

Jython 2.7.2 is distributed via an executable jar file installer. After downloading it, either double click the jython-installer-2.7.2.jar or run java with the -jar option

$ java -jar jython-installer-2.7.3.jar

This will start the regular GUI installer on most systems, or a console installer on headless systems. To force the installer to work in headless mode invoke the installer as:

$ java -jar jython-installer-2.7.2.jar --console

The installer will then walk through a similar set of steps in graphical or console mode: showing the license, selecting an install directory and JVM and actually copying Jython to the file system. After this completes, Jython is installed in the directory you selected. Executing a script in the install directory, jython on Unix-like systems or jython.exe on Windows, will start up the Jython console, which can be used to dynamically explore Jython and the Java runtime, or to run Jython scripts.


# Standalone Jar

The standalone option does no caching and so avoids the startup overhead (most likely at the cost of some speed in calling Java classes, but I have not profiled it)

You can try it out by running the installer:

$ java -jar jython-installer-2.7.2.jar

then when you come to the “Installation type” page, select “Standalone”.

The installation will generate a jython.jar with the Python standard library (/Lib) files included, which can be run as:

$ java -jar jython.jar

Of course you can run scripts just by calling them as you might expect:

$ java -jar jython.jar script.py

Or, add this file to the classpath of your application.

# Installation options

You can get a list of installer options (to install Jython unattended, for example) by running:

$ java -jar jython-installer-2.7.2.jar --help