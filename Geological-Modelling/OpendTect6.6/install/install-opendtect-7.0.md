# Դ�밲װOpendTect-7.0

wget -C https://github.com/OpendTect/OpendTect/archive/refs/tags/7.0.2.tar.gz

## Dependencies

To build the software you need to also download and install/build a few dependencies which probably are not installed in your system. The version of dependencies varies between the branches. The Qt dependencies are available in binary installers, the others have to be built from source.

BRANCH	DEPENDENCIES
main	Qt 5.15.2, OpenSceneGraph 3.6.5, Proj 9.2.0 (optional), Sqlite 3.40 (optional), HDF5 1.14.0 (optional)
od7.0_rel, od7.0	Qt 5.15.2, OpenSceneGraph 3.6.5, Proj 9.2.0 (optional), Sqlite 3.40 (optional), HDF5 1.14.0 (optional)
od6.6_rel, od6.6	Qt 5.15.2, OpenSceneGraph 3.6.5, HDF5 1.12.2 (optional)
od6.4.5, od6.4	Qt 5.9.6, OpenSceneGraph 3.6.3
Qt Install
For the Qt install the following components must be selected depending on your build platform:

Desktop msvc2019 64- bit (Windows), SDK 10.15 (macOS) or gcc 64 bit (Linux)
QtWebEngine
Optionally source code or debug information files
OpenSceneGraph Build
Configure using CMake, compile and install.

Proj Build
Configure using CMake, compile and install.

Sqlite Install
Retrieve from their download site or the OpendTect SDK

HDF5 Install
The link to HDF5 requires to provide the path to an existing HDF5 installation. All versions above 1.10.3 are supported, but using the current API 1.14 is preferred. Installation is best done using their binary installations (on Windows especially), or from the package manager on Linux. Windows developers however need to recompile the sources since no debug binary libraries can be downloaded.

