# MakefileÉú³É¶¯Ì¬¿â(.so)

```
# Make command to use for dependencies
RM=rm
MKDIR=mkdir
 
OUTDIR=_obj
SODIR=./bin
LIBNAME=pdk_ai.so
OUTFILE=$(SODIR)/$(LIBNAME)
INC=-I../../../../lib/svrlib/include -I../../../../lib/tinyxml -I../../../include
LIB=-lpthread -lrt -ldl
 
#obj dir
TINYXML=../../../../lib/tinyxml
EXEFILE=$(EXEDIR)/$(APP)
SRC := $(wildcard *.cpp $(TINYXML)/*.cpp)
OBJ := $(patsubst %.cpp, $(OUTDIR)/%.o, $(notdir ${SRC}))
     
     
COMPILE=g++ -c -o "$(OUTDIR)/$(*F).o" $(INC) -fPIC -w "$<"
LINK=g++ -o "$(OUTFILE)" $(OBJ) $(LIB) -shared -fPIC   # use -shared option!
 
# Pattern rules
$(OUTDIR)/%.o : $(TINYXML)/%.cpp
    $(COMPILE)
 
$(OUTDIR)/%.o : ./%.cpp
    $(COMPILE)
     
# Build rules
all: $(OUTFILE)
 
$(OUTFILE): $(OUTDIR)  $(OBJ)
    $(LINK)
#   sh sh_ver.sh ./win/svrlib.rc $(OUTFILE)   #update ver. ( read by "readelf -h libsvr.so")
#   sh sh_tar.sh  ./win/*.rc  $(SODIR)  $(LIBNAME) #tar file
 
$(OUTDIR):
    $(MKDIR) -p "$(OUTDIR)"
    $(MKDIR) -p "$(SODIR)"
     
# Rebuild this project
rebuild: cleanall all
 
# Clean this project
clean:
    $(RM) -f $(OUTFILE)
    $(RM) -f $(OBJ)
 
# Clean this project and all dependencies
cleanall: clean
```