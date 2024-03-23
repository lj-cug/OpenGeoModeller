project(
	'meson-cpp-template',
	'cpp', version: '0.1.0',
	license: 'MIT',
	default_options: ['cpp_std=c++20'],
	meson_version: '>= 1.0.0'
)

incdir = include_directories('include')

lib_sqrt = static_library('sqrt', 'lib/sqrt.cpp', include_directories: incdir)
dep_sqrt = declare_dependency(link_with: lib_sqrt)

lib_mymath = static_library('mymath', 'src/mymath.cpp', include_directories: incdir)
dep_mymath = declare_dependency(link_with: lib_mymath)

executable(
	'main',
	'src/main.cpp',
	include_directories: incdir,
	dependencies: [dep_sqrt, dep_mymath]
)