# git���ߵ�ʹ��ѧϰ

ʹ��git���߹������ֿ⣬�ο���Ŀ¼�µ�"git�̳�-LiuXuefeng"
�Լ�git-tutorials

# Gitee ������˵��

### 1����չ������ǰ��׼��

- ��װGit������ȷ�����ĵ������Ѿ���װ��Git������������˽�GIt��������Google��baidu��������������������������ݡ�
- �ڿ�չGitee�Ĺ�����֮ǰ������Ҫ����openEuler�Ĵ����й�ƽ̨�����ҵ�������Ȥ��Repository���������δ�ҵ���Ӧ��Repository����ο�[���½ڵ�����](README.md/#�ҵ�������Ȥ�Ĺ���)��



### 2��������fork�����֧

2. �ҵ����򿪶�Ӧ��repository����ҳ
2. ������Ͻǵ� `Fork` ��ť������ָ��������һ������**�����ˡ�**������fork��֧��<img src="figure/Gitee-workflow-fork.JPG" style="zoom:67%;" />



### 2.��fork��֧���Ƶ�����

�밴�����µĸ��ƹ��̽�repository�ڵĴ������ص������ڼ�����ϡ�

1��**�������ع���Ŀ¼**��

����Ҫ�������ع���Ŀ¼���Ա��ڱ��ش���Ĳ��Һ͹���

```
mkdir /YOUR_PATH/src/gitee.com/${your_working_dir}
```

> ˵����������Ѿ���openEuler������������ `XXX`���Ϊ�����е�`gitee.com` Ŀ¼����Ŀ¼.



2��**���git���û����������ȫ������**�������֮ǰ�Ѿ���ɹ��������ã�����ԣ�

��git�ϵ� `user` ���ó���gitee�ĸ������ƣ�

```
git config --global user.name "your Gitee Name"
```

��������git����

```
git config --global user.mail "email@your_Gitee_email"
```



3��**���SSH��Կע�ᣨ�����û����ɴ�ע�ᣬÿ�ζ�Ҫ���������˻������룩**

- �� ����ssh��Կ

  ```
  ssh-keygen -t rsa -C "email@your_Gitee_email"
  cat ~/.ssh/id_rsa.pub
  ```

- �� ��¼�����˵�Զ�ֿ̲���վGitee�˻����������ssh��Կ

  ����Gitee��ҳ������Ͻǵġ�����ͷ�񡱽������Gitee�˻������������ͷ���µġ��������á��������������ҳ�档�ڡ���������->��ȫ���á��£������SSH��Կ�����ڡ���ӹ�Կ���ڰ�cat�����ȡ����ssh��Կ��ӽ�ȥ��

  <img src="figure/Gitee-workflow-addSSHKey.JPG" style="zoom:67%;" />

  �ڸ��˵��������gitee��SSH�ϵĵǼ�

  ```
  ssh -T git@gitee.com
  ```

  ���������¡��ɹ�����ʾ�����ʾ ssh ��Կ�Ѿ���Ч��  
  `Hi $user_name! You've successfully authenticated, but GITEE.COM does not provide shell access.`
  

4��**����Զ�ֿ̲⵽����**

- �� **�л�������·��***

  ```
  mkdir -p $working_dir
  cd $working_dir
  ```

- �� **����Զ�ֿ̲⵽����**

  - ��ע��openEuler�м�����֯����ȷ���������ص�Զ�ֿ̲����֯����

  - ��������repository�ڸ���Զ�ֿ̲�Ŀ�����ַ���õ�`$remote_link`��

    <img src="figure/Gitee-workflow-CopyLink.JPG" alt="Gitee-workflow-CopyLink" />

  - �ڱ��ص���ִ�п������

    ```
    # ��Զ�� fork �ֿ⸴�Ƶ�����
    git clone https://gitee.com/$user_name/$repository_name

    # ���ñ��ع���Ŀ¼�� upstream Դ���� fork �����βֿ⣩
    git remote add upstream https://gitee.com/openeuler/$repository_name

    # ����ͬ����ʽ���˴�
    git remote set-url --push upstream no_push
    ```


### 3.����֧

�������ı��ط�֧

```
git fetch upstream
git checkout master
git rebase upstream/master
```

����������֧:

```
git checkout -b myfeature
```

Ȼ���� `myfeature` ��֧�ϱ༭���޸Ĵ��롣



### 4�����ع�������֤

���ع����ľ��巽������ο�����repository���ṩ������ĵ�����ȡ��ʽ��ο�[���½�](README.md/#id2-2-3)���ݡ�



### 5���������ķ�֧��master��ͬ��

```
# While on your myfeature branch
git fetch upstream
git rebase upstream/master
```

ִ��merge��ʱ���벻Ҫʹ�� `git pull` �������� `fetch` / `rebase`. `git pull` ����Ϊ���ַ�ʽ��ʹ�ύ��ʷ��û��ң���ʹ������ѱ���⡣��Ҳ����ͨ�������ļ����ﵽĿ�ģ� `.git/config` �ļ�ͨ�� `git config branch.autoSetupRebase always` ȥ�ı� `git pull`����Ϊ��



### 6���ڱ��ع���Ŀ¼�ύ���

�ύ���ı��

```
git add .
git commit -m "�ύԭ��"
```

�����ܻ���ǰ���ύ�Ļ����ϣ������༭���������Ը������ݣ�����ʹ�� `commit --amend` ��������ύ��



### 7�� ��������͵����Զ��Ŀ¼

׼��������飨��ֻ�ǽ�����������ر��ݣ�ʱ������֧�Ƶ�����`gitee.com`��fork��֧:

```
git push -f origin myfeature
```



### 8����Gitee�ϴ���һ�� pull request

1. �������� `https://gitee.com/$user/openEuler`��ҳ��

2. �����ķ�֧ѡ�����ύʹ�õ� `myfeature` ��֧�ϣ����`+ Pull Request` .����λ������ͼ��ʾ��

   <img src="figure/Gitee-workflow-PR1.JPG" style="zoom:80%;" />

3. �ڴ�����PR���棬ȷ��Դ��֧��Ŀ���֧��ѡ�񴴽���

4. �ύPR�Ƕ���Ŀ�ϵ�Master��һ�κ��룬Ϊ��֤����������������������ҪС�ľ����ģ������Բ鿴[pull-request](pull-request.md)�����ĵ��ж��ύPR�ĸ��Ӿ����ָ���ͽ��飬�԰������ύ��PR��ȷ�͸��ӿ��ٵĻ����Ӧ�ͺ���.

*���⣬�������������д����Ȩ*���벻Ҫʹ��Gitee UI����PR����ΪGitee�������洢�����������fork�д���PR��֧��



### 9���鿴�ͻ�Ӧ����������

#### �鿴����������

���ύPR�����PR�������һ�����������ߡ���Щ�����߽����г��׵Ĵ�����ӣ���ȷ���ύ����ȷ�ԣ����������������ȷ��Ҳ����ע�ͺ��ĵ��ȡ�

��������PR�б����ҵ����ύ��PR���������Ը�PR�����ۺ����������

![](figure/Gitee-workflow-PR2.JPG)

**С��PR�����׼��ӡ������ϴ��PR���ѱ���ȷ�ļ��ӡ�**



### ����һ���ύ

�����������ύ�����������ķ�ʽ

*�������������д����Ȩ��*���벻Ҫʹ��`Revert`Gitee UI�еİ�ť����PR����ΪGitee�������洢�����������fork�д���PR��֧��

- ����һ����֧����upstream����ͬ��

  ```
  # create a branch
  git checkout -b myrevert
  
  # sync the branch with upstream
  git fetch upstream
  git rebase upstream/master
  ```

- �����ϣ����ԭ���ύ��:

  - **merge commit:**

    ```
    # SHA is the hash of the merge commit you wish to revert
    git revert -m 1 SHA
    ```

  - **single commit:**

    ```
    # SHA is the hash of the single commit you wish to revert
    git revert SHA
    ```

- �⽫����һ���µ��ύ�Ի��˵�����ǰ�� push����ύ��Զ�̹���Ŀ¼

```
git push ${your_remote_name} myrevert
```

- �������֧����һ��PR.



### �����ύ��ͻ

����������ύ��PR�������µı�ǣ�˵�����ύ��PR�������ش��ڳ�ͻ������Ҫ�����ͻ��

![](figure/Gitee-workflow-confict.JPG)

1���Ƚ���֧�л���master�ϣ������master��rebase

```
git checkout master
git fetch upstream
git rebase upstream/master
```

2���ٽ���֧�л�����ʹ�õķ�֧�ϣ�����ʼrebase

```
git checkout yourbranch
git rebase master
```

3����ʱ��������git�Ͽ�����ͻ����ʾ�������ͨ��vi�ȹ��߲鿴��ͻ

4�������ͻ�Ժ��ٰ��޸��ύ��ȥ

```
git add .
git rebase --continue
git push -f origin yourbranch
```



# �ϲ��ύ

������ύ��һ��PR�Ժ󣬸��ݼ����������޸Ĳ��ٴ��ύ��PR���������������߿�������ύ��PR����Ϊ�ⲻ���ڼ����ڼ������޸ģ���ô�����Ժϲ��ύ��PR���ϲ��ύ��PR��ͨ��ѹ��Commit��ʵ�ֵġ�

1�����ڱ��ط�֧�ϲ鿴��־

```
git log
```

2��Ȼ��Ѷ�����n���ύ��¼�ۺϵ�һ����룬ע��n��һ�����֡�

```
git rebase -i HEAD~n
```

������ѹ������־ǰ���pick���ĳ�s��s��squash����д��ע����뱣��һ��pick����������е�pick���ĳ���s��û�кϲ���Ŀ���ˣ��ᷢ������

3���޸�����Ժ󣬰�ESC����������`:wq`��������һ�����棬�����Ƿ����༭�ύ��ע��ҳ�棬����e�Ժ󣬽���ϲ��ύ��ע��ҳ�档�����Ҫ�ϲ��ı�ע��ɾ����ֻ�����ϲ�Ŀ��ı�ע���ٰ�ESC��������`:wq`�����˳����ɡ�

4���������ύ

```
git push -f origin yourbranch
```

5���ص�gitee�ϵ�PR�ύҳ��鿴�����Ϳ��Կ���֮ǰ���ύ�Ѿ��ϲ��ˡ�
