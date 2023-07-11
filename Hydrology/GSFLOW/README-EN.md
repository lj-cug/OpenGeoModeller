

                      GSFLOW - Version: 1.2.0
          Coupled Groundwater and Surface-water FLOW model


NOTE: Any use of trade, product or firm names is for descriptive purposes 
      only and does not imply endorsement by the U.S. Government.

GSFLOW version 1.2.0 is packaged for personal computers using one of the 
Linux operating systems. The executable file was compiled using gfortran
(GNU Fortran) version 4.4.7 20120313 (Red Hat 4.4.7-16) on a computer running
Scientific Linux Release 6.7 (Carbon), Kernel Linux 2.6.32-573.3.1.el6.x86_64
with Intel Pentium E2160 CPU.

The source code and Linux Makefiles are provided to aid users in compilation
on other computers. However, no support is provided for compilation.

IMPORTANT: Users should review the file 'GSFLOW_Release_Notes.pdf' for a 
description of, and references for, this software. Changes that have been 
introduced into GSFLOW with each official release also are described in this 
file; these changes may substantially affect users.

Instructions for installation, execution, and testing of this version of
GSFLOW are provided below.



                            TABLE OF CONTENTS

                         A. DISTRIBUTION FILE
                         B. INSTALLING
                         C. EXECUTING THE SOFTWARE
                         D. TESTING
                         E. COMPILING


A. DISTRIBUTION FILE

The following compressed tar file is for use on personal computers:

         gsflow_v1.2.0.zip

The distribution file contains:

          Executable and source code for GSFLOW.
          GSFLOW documentation.
          Related documentation for PRMS, MODFLOW, and MODFLOW-NWT.
          Three GSFLOW example problems.
          Excel spreadsheets for analysis of GSFLOW results.

Extracting the distribution file creates numerous individual files
contained in several directories. The following directory structure
will be generated in the installation directory:

   |
   |--GSFLOW_1.2.0
   |    |--bin           ; Compiled GSFLOW executables for personal computers
   |    |--data          ; Three example GSFLOW application models described
                            in USGS reports TM6-D1 and TM6-D3.
   |    |--doc           ; Documentation reports for GSFLOW and related
                            software.
   |    |--src           
   |        |--gsflow    ; Source code for GSFLOW Modules
   |        |--mms       ; Source code for MMS software
   |        |--modflow   ; Source code for MODFLOW-2005 and MODFLOW-NWT 
                           Packages
   |        |--prms      ; Source code for PRMS Modules
   |    |--utilities     ; Utility program for analysis of GSFLOW output


It is recommended that no user files be kept in the GSFLOW_1.2.0 directory
structure.  If you do plan to put your own files in the GSFLOW_1.2.0
directory structure, do so only by creating additional subdirectories of
the GSFLOW_1.2.0/data subdirectory.

Included with the release are several documents that use the Portable Document 
Format (PDF) file structure. The PDF files are readable and printable on various 
computer platforms using Acrobat Reader from Adobe. The Acrobat Reader is freely 
available from the following World Wide Web site: http://www.adobe.com/


B. INSTALLING

To make the executable version of GSFLOW accessible from any directory, the 
directory containing the executable (GSFLOW_1.2.0/bin) should be included in the 
PATH environment variable. Also, if a prior release of GSFLOW is installed on your 
system, the directory containing the executables for the prior release should be 
removed from the PATH environment variable.
  
As an alternative, the executable files in the GSFLOW_1.2.0/bin directory 
can be copied into a directory already included in the PATH environment 
variable. The sample problem provided with the release (described below)
has sample batch files that provide an alternative, additional approach for
accessing the executable files.


C. EXECUTING THE SOFTWARE

A 64-bit (gsflow) executable is provided in the GSFLOW_1.2.0/bin directory.
After the GSFLOW_1.2.0/bin directory is included in your PATH, GSFLOW is
initiated in a Terminal window using the command:

      gsflow [Fname]

The optional Fname argument is the name of the GSFLOW Control File.  If 
no argument is used, then GSFLOW will look for a Control File named 
"control" in the user's current directory.

The arrays in GSFLOW are dynamically allocated, so models are not limited
by the size of input data. However, it is best to have at least 4 MB of 
random-access memory (RAM) for model execution and more RAM for large models.
If there is less available RAM than the model requires, which depends
on the size of the application, the program will use virtual memory; however,
this can slow execution significantly. If there is insufficient memory to 
run the model, then GSFLOW will not initiate the beginning of the simulation. 

Some of the files written by GSFLOW are unformatted files. The structure
of these files depends on the compiler and options in the code. For Linux
based computers, GSFLOW is compiled with the unformatted file type specified
as "UNFORMATTED". Any program that reads the unformatted files produced by
GSFLOW must be compiled with a compiler that produces programs that use the
same structure for unformatted files.  For example, Zonebudget and Modpath use 
unformatted budget files produced by the MODFLOW component of GSFLOW. Another 
example are head files that are generated by one GSFLOW simulation and used 
in a following simulation as initial heads. Both simulations must be run 
using an executable version of GSFLOW that uses the same unformatted file 
structure. Note: unformatted files produced on Linux are not usable on Windows-
based computers and vice versa.


D. TESTING

Three sample problems with GSFLOW data sets are provided in the 'data' sub-
directory to verify that GSFLOW is correctly installed and running on the 
user's system. The sample problems also may be looked at as examples of how 
to use the program. See the 'Readme.txt' file in that subdirectory for a 
description of the three sample problems.


E. COMPILING

The executable file provided in GSFLOW_1.2.0/bin was created using gfortran
and gcc compilers.  Although executable versions of the program are provided,
the source code also is provided in the GSFLOW_1.2.0/src directory so that
GSFLOW can be recompiled if necessary.  However, the USGS cannot provide
assistance to those compiling GSFLOW. In general, the requirements are a
Fortran compiler, a compatible C compiler, and the knowledge of using the
compilers. Makefiles are included in the GSFLOW_1.2.0\src directories as an
example for compiling GSFLOW.
