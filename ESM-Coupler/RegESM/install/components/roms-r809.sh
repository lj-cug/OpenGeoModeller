svn checkout -r 809 --username [USER NAME] https://www.myroms.org/svn/src/trunk roms-r809
cd roms-r809
wget https://raw.githubusercontent.com/uturuncoglu/RegESM/master/tools/ocn/roms-r809.patch
patch -p 3 < roms-r809.patch