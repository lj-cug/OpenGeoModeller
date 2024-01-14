
git客户端下载参考： git-for-Windows.md


git客户端安装参考：https://blog.csdn.net/fzx1597965407/article/details/124371720


什么是git
Git是一个开源的分布式版本控制系统，可以有效、高速地处理从很小到非常大的项目版本管理。也是Linus Torvalds为了帮助管理Linux内核开发而开发的一个开放源码的版本控制软件。

Git 是基于 Linux内核开发的版本控制工具。与常用的版本控制工具 CVS, Subversion 等不同，它采用了分布式版本库的方式，不必服务器端软件支持（wingeddevil注：这得分是用什么样的服务端，使用http协议或者git协议等不太一样。并且在push和pull的时候和服务器端还是有交互的。），使源代码的发布和交流极其方便。 Git 的速度很快，这对于诸如 Linux kernel 这样的大项目来说自然很重要。 Git 最为出色的是它的合并跟踪（merge tracing）能力。

git常用命令类型有：1、第一次初始化；2、工作基本操作；3、初始化仓库；4、查看仓库当前状态；5、文件相关操作；6、查看历史记录；7、代码回滚；8、版本库相关操作；9、远程仓库相关操作；10、分支相关操作；11、git相关配置；12、其他查看配置相关；13、撤消某次提交；14、标签。


1、第一次初始化
git init
git add .
git commit -m ‘first commit’
git remote add origin git@github.com:帐号名/仓库名.git
git pull origin master
git push origin master # -f 强推
git clone git@github.com:git帐号名/仓库名.git


2、工作基本操作
git checkout master 切到主分支
git fetch origin 获取最新变更
git checkout -b dev origin/master 基于主分支创建dev分支
git add . 添加到缓存
git commit -m ‘xxx’ 提交到本地仓库
git fetch origin 获取最新变更


3、初始化仓库
git init

4、查看仓库当前状态
git status

5、文件相关操作
将文件添加到仓库：

git add 文件名 将工作区的某个文件添加到暂存区
git add . 将当前工作区的所有文件都加入暂存区
git add -u 添加所有被tracked文件中被修改或删除的文件信息到暂存区，不处理untracked的文件
git add -A 添加所有被tracked文件中被修改或删除的文件信息到暂存区，包括untracked的文件
git add -i 进入交互界面模式，按需添加文件到缓存区
将暂存区文件提交到本地仓库：

git commit -m “提交说明” 将暂存区内容提交到本地仓库
git commit -a -m “提交说明” 跳过缓存区操作，直接把工作区内容提交到本地仓库
比较文件异同

git diff 工作区与暂存区的差异
git diff 分支名 工作区与某分支的差异，远程分支这样写：remotes/origin/分支名
git diff HEAD 工作区与HEAD指针指向的内容差异
git diff 提交id 文件路径 工作区某文件当前版本与历史版本的差异
git diff –stage 工作区文件与上次提交的差异(1.6 版本前用 –cached)
git diff 版本TAG 查看从某个版本后都改动内容
git diff 分支A 分支B 比较从分支A和分支B的差异(也支持比较两个TAG)
git diff 分支A…分支B 比较两分支在分开后各自的改动
另外：如果只想统计哪些文件被改动，多少行被改动，可以添加 –stat 参数

6、查看历史记录
git log 查看所有commit记录(SHA-A校验和，作者名称，邮箱，提交时间，提交说明)
git log -p -次数 查看最近多少次的提交记录
git log –stat 简略显示每次提交的内容更改
git log –name-only 仅显示已修改的文件清单
git log –name-status 显示新增，修改，删除的文件清单
git log –oneline 让提交记录以精简的一行输出
git log –graph –all –online 图形展示分支的合并历史
git log –author=作者 查询作者的提交记录(和grep同时使用要加一个–all–match参数)
git log –grep=过滤信息 列出提交信息中包含过滤信息的提交记录
git log -S查询内容 和–grep类似，S和查询内容间没有空格
git log fileName 查看某文件的修改记录

7、代码回滚
git reset HEAD^ 恢复成上次提交的版本
git reset HEAD^^ 恢复成上上次提交的版本，就是多个^，以此类推或用~次数
git reflog
git reset –hard 版本号
–soft：只是改变HEAD指针指向，缓存区和工作区不变；
–mixed：修改HEAD指针指向，暂存区内容丢失，工作区不变；
–hard：修改HEAD指针指向，暂存区内容丢失，工作区恢复以前状态；

8、版本库相关操作
删除版本库文件：git rm 文件名
版本库里的版本替换工作区的版本：git checkout — test.txt


9、远程仓库相关操作
同步远程仓库：git push -u origin master

本地仓库内容推送到远程仓库：git remote add origin git@github.com:帐号名/仓库名.git

从远程仓库克隆项目到本地：git clone git@github.com:git帐号名/仓库名.git

查看远程库信息：git remote

拉取远程分支到本地仓库：

git checkout -b 本地分支 远程分支 # 会在本地新建分支，并自动切换到该分支
git fetch origin 远程分支:本地分支 # 会在本地新建分支，但不会自动切换，还需checkout
git branch –set-upstream 本地分支 远程分支 # 建立本地分支与远程分支的链接
同步远程仓库更新：：git fetch origin master

10、分支相关操作
创建分支：git checkout -b dev  -b表示创建并切换分支
上面一条命令相当于一面的二条：
git branch dev  创建分支
git checkout dev  切换分支

查看分支：git branch

合并分支：

git merge dev #用于合并指定分支到当前分支
git merge –no-ff -m “merge with no-ff” dev #加上–no-ff参数就可以用普通模式合并，合并后的历史有分支，能看出来曾经做过合并
删除分支：git branch -d dev

查看分支合并图：git log –graph –pretty=oneline –abbrev-commit

11、git相关配置
安装完Git后第一件要做的事，设置用户信息(global可换成local在单独项目生效)：

git config –global user.name “用户名” # 设置用户名
git config –global user.email “用户邮箱” #设置邮箱
git config –global user.name # 查看用户名是否配置成功
git config –global user.email # 查看邮箱是否配置


12、其他查看配置相关
git config –global –list # 查看全局设置相关参数列表
git config –local –list # 查看本地设置相关参数列表
git config –system –list # 查看系统配置参数列表
git config –list # 查看所有Git的配置(全局+本地+系统)
git config –global color.ui true //显示git相关颜色


13、撤消某次提交
git revert HEAD # 撤销最近的一个提交
git revert 版本号 # 撤销某次commit


14、标签
git tag 标签 //打标签命令，默认为HEAD
git tag //显示所有标签
git tag 标签 版本号 //给某个commit版本添加标签
git show 标签 //显示某个标签的详细信息
