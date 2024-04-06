# build oasis3-mct-3.0
cd oasis3-mct/util/make_dir
## Adapt the "make.inc" file to include your platform header makefile
## Adapt the value of $COUPLE and $ARCHDIR in your platform header makefile
make realclean ¨Cf TopMakefileOasis3
make ¨Cf TopMakefileOasis3
## The libraries "libmct.a", "libmpeu.a", "libpsmile.MPI1.a" and "libscrip.a" that need to
##   be linked to the models are available in the directory $ARCHDIR/lib

## Compile tutorial
cd oasis3-mct/examples/tutorial
make clean; make
./run_tutorial

#--------------------------------------------------------------
# build oasis3-mct-4.0
cd oasis3-mct-4.0\util\make_dir
make make_ubuntu
