# ����ROMSԴ��
## ע���˻�
```
Username: lijian1984
Password: LIjian1984
```

## SVN��������

svn co --username [YOUR_USER_NAME] https://www.myroms.org/svn/src/tags/roms-3.5 roms-3.5

## ��ϸ����������

To check-out the files from the ROMS repository trunk (latest version), enter (notice https instead of http):

   svn checkout https://www.myroms.org/svn/src/trunk MyDir 
   
where MyDir is the destination directory on your local computer. It will be created if not found. If your username on your local computer is not the same as your ROMS username you will need to pass the --username option to svn:

   svn checkout --username joe_roms https://www.myroms.org/svn/src/trunk MyDir
   
Files for test cases and other repositories

There is an additional repository that provides idealized Test Cases and realistic applications to introduce you to ROMS many capabilities. You may use these examples as a guideline in setting up your own application. Currently, this repository requires ~388MB of disk space. This is because the realistic applications require several input NetCDF files. To checkout this repository, enter:

   svn checkout https://www.myroms.org/svn/src/test MyTest
   
Similarly, to check-out the files from the ROMS repository matlab (official scripts), enter:

   svn checkout https://www.myroms.org/svn/src/matlab MyDir
   
To check-out the files from the ROMS repository plot (NCAR's library-based plotting package), enter:

   svn checkout https://www.myroms.org/svn/src/plot MyDir  
   
To check-out the files from the ROMS repository branches, plot, matlab, tags, test, and trunk, (this is not recommended because it takes a long time to complete) enter:

   svn checkout https://www.myroms.org/svn/src MyDir
   
You only check out once, after that, a hidden directory called .svn exists to keep track of the source, destination and a bunch of other information. Your username and password will also be saved. For more detail on command line use and syntax, see the svn book.

Several GUI front-ends to subversion exist, allowing the user to have visual prompts to help them manage their files.