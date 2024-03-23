# Meson C++ Project Template

This is a template for a C++ project using Meson. It is intended to be used as a starting point for new projects.

## Features

* C++20
* CLang
* Meson 1.0.0
* Build Script
* Dependency Checking Script

## Usage

>Note: Before running build.py without calling it with python you need to make it executable by running `chmod +x build.py`
1. Clone this repository
2. Run `build.py -m` to configure the build environment.
3. Run `build.py -n` to build the project.
4. Run `build.py -c` to clean the build directory.

## Dependencies

* Meson 1.0.0+
* Ninja
* Python 3.6+
* Pip
* C++ Compiler (GCC, CLang, MSVC) with C++20 support (I use CLang)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
