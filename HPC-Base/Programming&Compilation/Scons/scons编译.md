# Samoa��SCons����

vars = Variables()

ͨ��Variables������չscons֧�ֵ������в��������Ʋ������ࣨBool��ö�٣���ѡ�������԰��ϴεĲ������浽�ļ����´α��벻��Ҫ�������������Ĳ�����

vars.AddVariables(
PathVariable( 'config', 'build configuration file', None, PathVariable.PathIsFile),
)

env = Environment(variables=vars)

Environment���Ա���ȫ�׵ı�������������ã���Clone�������Դ�һ���������뻷�������ܶ���֣�֧�ֲ�ͬ��Ŀ���ͣ���ͬ���õı��롣Cloneʵ����̫�����ˡ�������ǰ���ᵽ��Windowsƽ̨�±���DLL�;�̬�⡢�����г���Ĳ�ͬ�������ã���Clone���������ڻ������뻷���ϣ����ɶ�����ڱ��뾲̬�⡢DLL�Ȳ�ͬ�ı��뻷����ʹ�ò�ͬ�Ĳ������롣

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
