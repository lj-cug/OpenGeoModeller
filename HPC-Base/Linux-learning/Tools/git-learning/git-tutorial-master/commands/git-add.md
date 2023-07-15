# git add

## 概述

`git add`命令用于将变化的文件，从工作区提交到暂存区。它的作用就是告诉 Git，下一次哪些变化需要保存到仓库区。用户可以使用`git status`命令查看目前的暂存区放置了哪些文件。

```bash
# 将指定文件放入暂存区
$ git add <file>

# 将指定目录下所有变化的文件，放入暂存区
$ git add <directory>

# 将当前目录下所有变化的文件，放入暂存区
$ git add .
```

## 参数

`-u`参数表示只添加暂存区已有的文件（包括删除操作），但不添加新增的文件。

```bash
$ git add -u
```

`-A`或者`--all`参数表示追踪所有操作，包括新增、修改和删除。

```bash
$ git add -A
```

Git 2.0 版开始，`-A`参数成为默认，即`git add .`等同于`git add -A`。

`-f`参数表示强制添加某个文件，不管`.gitignore`是否包含了这个文件。

```bash
$ git add -f <fileName>
```

`-p`参数表示进入交互模式，指定哪些修改需要添加到暂存区。即使是同一个文件，也可以只提交部分变动。

```bash
$ git add -p
```

注意，Git 2.0 版以前，`git add`默认不追踪删除操作。即在工作区删除一个文件后，`git add`命令不会将这个变化提交到暂存区，导致这个文件继续存在于历史中。Git 2.0 改变了这个行为。

## 实现细节

通过`git add`这个命令，工作区里面那些新建或修改过的文件，会加入`.git/objects/`目录，文件名是文件内容的 SHA1 哈希值。`git add`命令同时还将这些文件的文件名和对应的哈希值，写入`.git/index`文件，每一行对应一个文件。

下面是`.git/index`文件的内容。

```
data/letter.txt 5e40c0877058c504203932e5136051cf3cd3519b
```

上面代码表示，`data/letter.txt`文件的哈希值是`5e40c087...`。可以根据这个哈希值到`.git/objects/`目录下找到添加后的文件。
