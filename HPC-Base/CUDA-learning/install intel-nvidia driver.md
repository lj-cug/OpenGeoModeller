# Intel for display, NVIDIA for computing

This guide will show you how to use Intel graphics for rendering display and NVIDIA graphics for CUDA computing on Ubuntu 18.04 / 20.04 desktop.

I made this work on an ordinary gaming PC with two graphics devices, an Intel UHD Graphics 630 plus an NVIDIA GeForce GTX 1080 Ti.
Both of them can be shown via `lspci | grep VGA`.

```plain
00:02.0 VGA compatible controller: Intel Corporation Device 3e92
01:00.0 VGA compatible controller: NVIDIA Corporation GP102 [GeForce GTX 1080 Ti] (rev a1)
```

This guide can be summarized into two steps:

1.  To install NVIDIA drivers without OpenGL files.
2.  To configure Xorg to use the Intel graphics.

I haven't tested on different hardware configurations, but it should work similarly.
See [this section](#Discussion) for more discussion.

## 0. Preparation

Before operating within Linux, you need to make some configuration on your hardware.
Make sure monitors are plugged to the motherboard the instead of dedicated display card.
Configure the BIOS to make Intel graphics as the primary display device (usually select IGFX instead of PEG or PCIE).
Make sure your computer could boot to GUI and be logged in to desktop successfully under this setting.

## 1. Install NVIDIA driver without OpenGL files

I suggest installing the driver in either of the following ways.
If you would like to follow your own method, just make sure the OpenGL files are not installed.

### 1.1. Uninstall all previous installations

Common steps for both methods to avoid possible conflicts.

1.  If you have installed via PPA repositories

    ```bash
    sudo apt purge nvidia*
    # some interactive operations
    sudo apt autoremove
    # some interactive operations
    ```

    Check remaining packages related to NVIDIA.

    ```bash
    dpkg -l | grep nvidia
    ```

    If some packages are not removed, manually remove them by

    ```bash
    sudo dpkg -P <package_name>
    ```

    If you have add third party repositories, e.g. ones from NVIDIA, remove them too.
    This could be done by removing related files under `/etc/apt/source.list.d/` or via `ppa-purge` utility.

2.  If you have installed via binary installers

    ```bash
    sudo nvidia-uninstall
    # some interactive operations
    ```

3.  Reboot.

### 1.2.A. Install from PPA Repository

1.  Add the `ppa:graphics-drivers/ppa` repository.

    ```bash
    sudo add-apt-repository ppa:graphics-drivers/ppa
    # some interactive operations
    ```

    On ubuntu 18.04 `sudo apt update` is automatically performed after adding a PPA, thus not manually required.

2.  Install the **headless** NVIDIA driver

    ```bash
    sudo apt install nvidia-headless-418 nvidia-utils-418
    ```

    Version 418 is the latest when I write this page. Changing to the latest version available is a good idea.

    **IMPORTANT**

    The `nvidia-headless-418` contains only the driver, while the full `nvidia-driver-418` package contain everything including display component such OpenGL libraries. If you hope to connect the display to a NVIDIA display card, install the full package, otherwise, install only the driver.

    The `nvidia-utils-418` package provide utilities such as `nvidia-smi`.

3.  Reboot. If the installation is successful, command `nvidia-smi` will show all NVIDIA GPUs.

### 1.2.B. Install from Binary Installer

[My previous post](https://gist.github.com/wangruohui/df039f0dc434d6486f5d4d098aa52d07#creat-blacklist-for-nouveau-driver), thought old, is detailed and still work.

To summary:

1.  Download the binary installer from [NVIDIA official site](https://www.nvidia.com/object/unix.html) and make it executable.

2.  Disable nouveau driver by creating file `/etc/modprobe.d/blacklist-nouveau.conf` with content

    ```plain
    blacklist nouveau
    options nouveau modeset=0
    ```

    or just executing the installer and it will prompt to create it for you.

3.  Execute `sudo update-initramfs -u` and then reboot.

4.  After booting, switch to tty1. Stop display services such as `gdm`, `gdm3`, `lightdm`. Kill all process that is using the graphic card, such as `Xorg` or `X`.

5.  Execute the installer with `--no-opengl-files` suffix (**IMPORTANT**) to avoid installation of all OpenGL related files, like

    ```bash
    sudo ./NVIDIA-Linux-x86_64-418.56.run --no-opengl-files
    ```

    Or if you would like to display from an NVIDIA graphic card, execute the installer without any arguments, like

    ```bash
    sudo ./NVIDIA-Linux-x86_64-418.56.run
    ```

6.  After a successful installation, command `nvidia-smi` will show all NVIDIA GPUs.

## 2. Configure Xorg

The installed NVIDIA driver and configurations will hint Xorg to start with NVIDIA devices.
Depending on whether NVIDIA related display libraries are well installed, the X server would failed to start or success to start but still use NVIDIA devices, both of which are unwanted.

We can force Xorg to use Intel graphics by creating a configuration file with the following contents and save it to `/etc/X11/xorg.conf`.

```plain
Section "ServerLayout"
    Identifier     "Layout0"
    Screen      0  "Screen0"
EndSection

Section "Screen"
    Identifier     "Screen0"
    Device         "Device0"
EndSection

Section "Device"
    Identifier     "Device0"
    Driver         "intel"
    VendorName     "Intel Corporation"
    BusID          "PCI:0:2:0
EndSection
```

The key point is the "BusID" option. It indicates the PCI bus id that the Intel graphics connects to.
It can be retrieved from `lspci`.
For example, on my computer, `lspci` outputs `00:02.0 VGA compatible controller: Intel Corporation Device 3e92`, thus the bus id is `0:2:0`.

Note that the bus id output from `lspci` is hexadecimal but the number filled in `xorg.conf` should be decimal.
For example, if the output from `lspci` is `82:00.0 VGA com...`, you need to fill `PCI:130:0:0` in the configuration.

On Ubuntu 20.04 you may want to set the driver to `modesetting` instead of `intel`. I met some problem and solved as [this link](https://askubuntu.com/questions/1231824/fuzzy-graphics-after-upgrading-to-ubuntu-20-04) (and links within it) describes.

Setting with multiple monitors would be a little bit complex.
Here is my example of setting with two monitors. Some fields are missing but it works as Xorg will smartly use some default configs.
Anyway, search the Internet to get a proper set of configurations for you. `man xorg.conf` and [ArchLinux wiki](https://wiki.archlinux.org/index.php/Xorg) are good references.

```plain
Section "Device"
    Identifier     "Device0"
    Driver         "modesetting"
    VendorName     "Intel Corporation"
    BusID          "PCI:0:2:0
    Option         "TearFree" "true"
    Option         "monitor-DP-1" "DP"
    Option         "monitor-HDMI-2" "HDMI"
EndSection

Section "Monitor"
    Identifier "DP"
    Option     "Rotate" "Left"
EndSection

Section "Monitor"
    Identifier "HDMI"
    Option     "RightOf " "DP"
    Option     "Primary" "true"
EndSection
```

After the configuration, reboot the computer.
If it successes, you will be able to login to the desktop.
If it fails, you could be locked at the login screen.
Reboot to advance mode, drop to root prompt and check `/var/log/Xorg.*.log.*` for hints.

After login successfully, open an terminal and execute `glxheads`.
The displayed rendering devices should be the Intel graphics.

```plain
glxheads: exercise multiple GLX connections (any key = exit)
Usage:
  glxheads xdisplayname ...
Example:
  glxheads :0 mars:0 venus:1
Name: :0
  Display:     0x56151a1570e0
  Window:      0x2000002
  Context:     0x56151a182f60
  GL_VERSION:  3.0 Mesa 18.0.5
  GL_VENDOR:   Intel Open Source Technology Center
  GL_RENDERER: Mesa DRI Intel(R) HD Graphics (Coffeelake 3x8 GT2)
```

Check `nvidia-smi`, there should be no `Xorg` process listed and memory occupation should be `0MB`.

```plain
Thu Nov 22 07:05:55 2018
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.56       Driver Version: 418.56       CUDA Version: 10.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  Off  | 00000000:01:00.0 Off |                  N/A |
|  0%   52C    P0    57W / 250W |      0MiB / 11178MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

Execute a heavy GPU program and move your mouse during the computation.
The display should be smooth, as the computing and display rendering are separated on different devices.

Congratulations!

If `glxheads` outputs some virtual graphics and `nvidia-smi` still output `Xorg` processes, you probably still using the NVIDIA graphics for rendering. It just pass the rendered image to Intel graphics for display. When running heavy GPU programs, the display will still be slow.

## 3. Discussion

Separating display and computing is important for a smooth work.

1.  You could use a remote headless server for computing a local client for display. You can connect the remote server via SSH. This could be the simplest way. Under this setting, a properly installed headless driver will make everything work. You don't need to follow this guide.

2.  If you have only one computer and there is only one graphic device in your computer. Its unfortunate. You have to use the only device for all tasks and you would suffer severe display latency when running GPU programs.

3.  You user a single computer for both computing and display. There are two (or more) graphic devices and they are from the same vendor. Then the problem could be mush easier. Taking NVIDIA as the example, with a properly installed driver, display and computing can be perfectly separated. One can plug monitors to device 0 and use `CUDA_VISIBIE_DEVICE=1` environment flag to perform computing on device 1. You probably don't need this guide.

4.  You user a single computer for both computing and display. There are two (or more) graphic devices but they are from different vendors. For example, an ordinary gaming PC configuration could include a Intel HD Graphics within the CPU and a dedicated GPU from NVIDIA. One need to plug monitors to the motherboard to use the Intel one for display and run CUDA program on the NVIDIA one. Then you are at the right place.

5.  For some purpose, you need an NVIDIA GPU for both computing and rendering. Then you probably need to run both of them on a single GPU. You need to install the drvier with OpenGL support and tune some rendering settings in Xorg.
