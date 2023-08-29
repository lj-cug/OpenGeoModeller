# Samoa的SCons编译

vars = Variables()

通过Variables可以扩展scons支持的命令行参数，限制参数种类（Bool，枚举，多选），可以把上次的参数保存到文件，下次编译不需要命令行输入过多的参数。

vars.AddVariables(
PathVariable( 'config', 'build configuration file', None, PathVariable.PathIsFile),
)

env = Environment(variables=vars)

Environment可以保存全套的编译命令，参数配置，用Clone操作可以从一个基本编译环境作出很多变种，支持不同项目类型，不同配置的编译。Clone实在是太方便了。下面是前面提到的Windows平台下编译DLL和静态库、命令行程序的不同参数配置，用Clone操作可以在基本编译环境上，生成多个用于编译静态库、DLL等不同的编译环境，使用不同的参数编译。

#Set config variables from config file if it exists
if 'config' in env:
  vars = Variables(env['config'])
else:
  vars = Variables()

#Add config variables
vars.AddVariables(
  PathVariable( 'config', 'build configuration file', None, PathVariable.PathIsFile),

  PathVariable( 'build_dir', 'build directory', 'bin/', PathVariable.PathIsDirCreate),
  
......

)
