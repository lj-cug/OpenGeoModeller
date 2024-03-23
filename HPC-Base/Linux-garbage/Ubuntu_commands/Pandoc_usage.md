# PANDOC

Markdown����ƽʱ��д������Ϻ�����ʱ��ϲ���Ĺ���֮һ����������Ϊ�ĵ��ַ�ʱȴ����Щ���㡣�Ͼ����������˶��˽� Markdown ���﷨���������಻�縻�ı��ĵ��׶���ֱ���ҷ�����Pandoc�������Խ��ĵ��� Markdown��LaTeX��reStructuredText��HTML��Word docx �ȶ��ֱ�Ǹ�ʽ֮���໥ת������֧����� PDF��EPUB��HTML �õ�Ƭ�ȶ��ָ�ʽ���ó��򱻳�Ϊ��ʽת����� ����ʿ��������

��ƽʱ����ʹ�øó������� HTML �� PDF �ĵ���ϣ�����Ŀ����ܽ�һЩ�Լ�ʹ�õ��ĵá�

## ������

Pandoc ��������John MacFarlane�����Ǽ��ݴ�ѧ��������У����ѧϵ���ڡ�Pandoc ʹ��Haskell���Ա�д���������������ɽ��塢�μ�����վ�ȡ��ó���Դ��ѣ�Ŀǰ�� GPL Э���й���Github ��վ�ϡ�

## ����װ

Pandoc �İ�װ�����෽ʽ��������ֻ������򵥵ķ�����Pandoc �������Ѿ�Ϊ Windows��macOS��Linux �Ȳ���ϵͳ�ֱ������˶�Ӧ�ĳ���װ����Ҫʹ�øó���ֻ�����ض�Ӧ�ĳ���װ�����а�װ���ɡ�

���� Ubuntu �� Linux ���а棬Pandoc �Ѿ������ɵ�ϵͳ������Դ�ڣ���˻�����ֱ�Ӵ�����Դ��װ��

sudo apt-get install pandoc

���ߣ�������Ѿ���װ�� Anaconda����ô�����ֱ��ʹ�� Pandoc �ˡ��ó����Ѿ������ɵ� Anaconda �С�

## ����˵��

Pandoc ���������ʹ�÷�ʽΪ��

pandoc <files> <options>

����<files>Ϊ��������ݣ������뼴���������ļ���Ҳ�������Ա�׼����������ҳ���ӡ���<options>Ϊ����ѡ���Ҫ�Ĳ���ѡ���У�

-f <format>��-r <format>��ָ�������ļ���ʽ��Ĭ��Ϊ Markdown��

-t <format>��-w <format>��ָ������ļ���ʽ��Ĭ��Ϊ HTML��

-o <file>��ָ������ļ�������ȱʡʱ�����������׼�����

--highlight-style <style>�����ô���������⣬Ĭ��Ϊ?pygments��

-s��������ͷβ�Ķ����ļ���HTML��LaTeX��TEI �� RTF����

-S������ģʽ�������ļ��ж����ʽ��

--self-contained�������԰������ļ���������� HTML �ĵ�ʱ��Ч��

--verbose������ Verbose ģʽ������ Debug��

--list-input-formats���г�֧�ֵ������ʽ��

--list-output-formats���г�֧�ֵ������ʽ��

--list-extensions���г�֧�ֵ� Markdown ��չ������

--list-highlight-languages���г�֧�ִ�������ı�����ԣ�

--list-highlight-styles���г�֧�ֵĴ���������⣻

-v��--version����ʾ����İ汾�ţ�

-h��--help����ʾ����İ�����Ϣ��

��Ȼ Pandoc �ṩ������ָ�����������ʽ�Ĳ��������Ǻܶ�ʱ��ò�������ʹ�á�Pandoc �Ѿ��㹻���������Ը����ļ����ж����������ʽ�����Գ����ļ�������������壬��������������������ʡ�ԡ�

## ʹ��ʾ��

### ��Ϣ�鿴

�鿴����֧�ֵ������ļ���ʽ��

$ pandoc --list-input-formats

�鿴����֧�ִ�������ı�����ԣ�

pandoc --list-highlight-languages

�鿴���������

pandoc --help

���� HTML �ĵ�

ʹ�� Pandoc ���Ժ����׵ؽ� Markdown �ĵ���ȾΪ HTML ��ҳ��

pandoc demo.md -o demo.html

�����������һ�� HTML �ĵ��������ĵ��������κ���ʽ��������ʾЧ����������ʹ�õ�����������ǵ�Ȼϣ�����Եõ��Ű���������ĵ���ֻҪ��ת��ʱ�����Լ��Ĳ����ʽ�� CSS �ļ�������� CSS �ļ���ʹ��-c������ָ����

pandoc demo.md -c style.css -o demo.html

�������� HTML �ĵ��Ѿ�������ʽ�ĵ��ˣ�ƽʱ�Լ��鿴ʱ��Ч���ܲ��������÷�ʽ��Ȼ���ڲ������⡣���Ƿ��������ĵ�ʱ����Ҫ�������������ļ���1 �� HTML �ļ��� 1 �� CSS �ļ�������Щ���㡣������ĵ��л������������ͼƬ���ļ��������ĵ��������˲����ܵ����顣���� Pandoc ���Խ��ⲿ�ļ�Ƕ�뵽 HTML �ĵ��У�����һ���԰����Ķ����ļ���

pandoc demo.md --self-contained -c style.css -o demo.html

�ڸ������У�--self-contained?����ָ�������κε��ⲿ�ļ�Ƕ����������ļ��У��γ�һ�������� HTML �ĵ���������������ʱֻ����һ���ļ��Ϳ����ˣ�������� PDF �ĵ�һ�����㡣

### ���� docx �ĵ�

��Ȼ�Һ�ϲ��ʹ�� HTML ��Ϊ�ĵ�������ʽ����ĳЩ���������ܻ�����Ҫ���� Word docx �ļ�����Ҳ�������⣬Pandoc �ܹ�����֧�ֵ������ļ�һ��ת��Ϊ Word docx ��ʽ��

��������һ�� Markdown �ļ�ת��Ϊ docx ��ʽ��

pandoc demo.md -o demo.docx

�������� HTML ��ҳת��Ϊ docx ��ʽ��

pandoc http://gnss.help/2017/06/12/pandoc-install-usage/ -o this_page.docx

�貹����ǣ�Pandoc �޷�Ϊ���ɵ� Word docx �ĵ�ָ���Ű淽ʽ���������Ҫ���α༭������ļ��������⡢���ĵȵ���Ϊ�������ʽ��

### ���� PDF �ĵ�

ʹ�� Pandoc ֱ������ PDF �ļ�ʱ����Ҫ��װ LaTeX�����ң�Pandoc �Դ��� PDF ���治֧�����ģ�����Ϊ�������ö���������ģ�塣Pandoc �������� PDF �ļ�������Ϊ��

pandoc demo.md -o demo.pdf

������ PDF �ĵ�ʱ��δʹ�����ϵķ��������ǲ��� HTML �ļ���Ϊ�м��ļ����ɣ�ʹ�� Windows ϵͳ�� ����ӡ�� PDF�� ���ܣ��� HTML �ĵ���һ��ת��Ϊ����� PDF �ĵ���

### ���� Markdown �ĵ�

������ Markdown Ҳ�� Pandoc ֧�ֵ������ʽ֮һ�����ǿ��Խ��κ�֧�ֵ������ʽת��Ϊ Markdown����������ǽ�֮ǰ���ĵ�Ҳ�л��� Markdown ��ʽ��˵��ʵ����̫�����ˡ�

����������� Word docx �ĵ����� Markdown �ļ���

pandoc demo.docx -o demo.md

����������� HTML ��ҳ���� Markdown �ĵ���

pandoc http://gnss.help/2017/06/12/pand

�����ʽ���ļ��������յ���ʾ��ʽ�������һ��Ư���� CSS ��ʽ���ļ��ǳ���Ҫ���ڴ��Ƽ����� CSS �ļ����������� Alberto Leal ������?Github ������ʽ���ļ���������ʾЧ�������� Github ��վ�� README �ĵ�����һ�����������ģ����Ʊ�վ�����õ�?Minos?���⣨Minos-style������ʽ���ļ������ļ���δ��ȫ�ȶ������貿�����ƣ������Ը�һ��������Ű��Ѿ�û�����⡣

# Compiling from source

If for some reason a binary package is not available for your platform, or if you want to hack on pandoc or use a non-released version, you can install from source.

Getting the pandoc source code

Source tarballs can be found at?https://hackage.haskell.org/package/pandoc. For example, to fetch the source for version 1.17.0.3:

wget https://hackage.haskell.org/package/pandoc-1.17.0.3/pandoc-1.17.0.3.tar.gz

tar xvzf pandoc-1.17.0.3.tar.gz

cd pandoc-1.17.0.3

Or you can fetch the development code by cloning the repository:

git clone https://github.com/jgm/pandoc

cd pandoc

Note: there may be times when the development code is broken or depends on other libraries which must be installed separately. Unless you really know what you��re doing, install the last released version.

## Quick stack method

The easiest way to build pandoc from source is to use?stack:

Install stack. Note that Pandoc requires stack >= 1.6.0.?

wget -qO- https://get.haskellstack.org/ | sh

Change to the pandoc source directory and issue the following commands:

## stack setup

stack install

stack setup?will automatically download the ghc compiler if you don��t have it.?stack installwill install the?pandoc?executable into?~/.local/bin, which you should add to your?PATH. This process will take a while, and will consume a considerable amount of disk space.

pandoc -f latex -t docx -o test.docx test.tex