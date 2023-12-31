# 常用连接Linux的SSH工具、SFTP工具

## 一、SSH工具

### 1.1 SecureCRT

SecureCRT是一款支持SSH(SSH1和SSH2)的终端仿真程序，简单地说是Windows下登录UNIX或Linux服务器主机的软件。  

SecureCRT支持SSH，同时支持Telnet和rlogin协议。SecureCRT是一款用于连接运行包括Windows、UNIX和VMS的理想工具。通过使用内含的VCP命令行程序可以进行加密文件的传输。有流行CRTTelnet客户机的所有特点,包括:自动注册、对不同主机保持不同的特性、打印功能、颜色设置、可变屏幕尺寸、用户定义的键位图和优良的VT100,VT102,VT220和ANSI竞争.能从命令行中运行或从浏览器中运行.其它特点包括文本手稿、易于使用的工具条、用户的键位图编辑器、可定制的ANSI颜色等.SecureCRT的SSH协议支持DES,3DES和RC4密码和密码与RSA鉴别。

### 1.2 XShell

Xshell是一个强大的安全终端模拟软件，它支持SSH1, SSH2, 以及Microsoft Windows 平台的TELNET 协议。Xshell 通过互联网到远程主机的安全连接以及它创新性的设计和特色帮助用户在复杂的网络环境中享受他们的工作。

### 1.3 Putty

PuTTY是一个Telnet、SSH、rlogin、纯TCP以及串行接口连接软件。较早的版本仅支持Windows平台，在最近的版本中开始支持各类Unix平台，并打算移植至Mac OS X上。除了官方版本外，有许多第三方的团体或个人将PuTTY移植到其他平台上，像是以Symbian为基础的移动电话。PuTTY为一开放源代码软件，主要由Simon Tatham维护，使用MIT licence授权。

缺点：

1、不支持标签模式；
2、默认设置不友好，很多功能都需要额外配置才行，例如自动登录功能；
3、不能传输文件；
4、没有X11，需要配置Xming工具；
5、默认keepalives没有设置，一段时间不操作后会断开。

## 二、SFTP工具

### 2.1 WinSCP

WinSCP是一个支持SSH的SCP文件传输软件。WinSCP中文版体积小、占用系统资源少。操作简单，只需要连接相应的服务器就可以进行下载和传输文件。重要的是WinSCP中文版软件还有着很多特色的功能，有着内置的文本编辑器，可以支持文件的复制、移动、更名文件等操作，为你带来高效便捷的使用体验。

### 2.2 FileZilla

FileZilla 客户端是一个快速可靠的、跨平台的FTP,FTPS和SFTP客户端。具有图形用户界面(GUI)和很多有用的特性。

### 2.3 Xftp

是一个基于 MS windows 平台的功能强大的SFTP、FTP 文件传输软件。使用了 Xftp 以后，MS windows 用户能安全地在 UNIX/Linux 和 Windows PC 之间传输文件。Xftp 能同时适应初级用户和高级用户的需要。它采用了标准的 Windows 风格的向导，它简单的界面能与其他 Windows 应用程序紧密地协同工作，此外它还为高级用户提供了众多强劲地功能特性。
