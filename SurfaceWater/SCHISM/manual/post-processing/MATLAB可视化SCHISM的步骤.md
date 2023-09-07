**使用步骤：**

1.  Add path:

将 ./ selfeutility 路径添加到matlab

（2）将matlab工作路径调整到./selfe_matlab

（3）演示

slf=selfe(\' D:\\SCHISM_Usages\\Utility\\SELFE_MATLAB\\Example\');

figure;

trisurf(slf.elem,slf.x,slf.y,slf.elev(1,:)\');

**selfe.m的帮助：**

\% obj=selfe(DirIn) - A netcdf style data class to assist reading

\% SELFE output data from \"DirIn\", for matlab 2008a (and later
versions).

\% The constructed object is a wrapper for the \"sz_readHeader\" and

\% \"sz_readTimeStep\" scripts provided in m-elio and require data in a
binary

\% Data Format v5.00 (sz-levels) and offers simple alternative to
indexing

\% output files.

\%

\% Index references are strictly in this order:

\% time, element, vertical layer, u/v

\% e.g. selfeobj.hvel(2,1,1,1) would access u velocity for the 2nd
timestep, 1st element and the bottom layer.

\% Note the following important information:

\% hgrid.gr3, vgrid.in and param.in\* files must be present in
\"DirIn\", as well output data.

\% \* in \"param.in\" start date needs to be at the start of the param
file, 1st two lines of the

\% param file may look like this for e.g.

\% ! Note require start time at start of param file for selfe data
object\...

\% ! 20/10/2010 10:00

\%

\% The directory \"DirIn\" the must be clean with only combined output
files ,

\% (and hgrid.gr3, vgrid.in, param.in) and no missing files (i.e. file
1,2,3,4\... must all be present).

\% Uncombined files should be moved to a separated directory within or
outside of the \"DirIn\" directory.

\% Note that filenames and date format (\"dd/mm/yyyy HH:MM\" format) are
hardwired in \"/selfeutility/initSELFEvar.m\" as:

\% result.param_file=\'param.in\';

\% result.grid_file=\'hgrid.gr3\';

\% result.vgrid_file=\'vgrid.in\';

\% result.dateFormat=\'dd/mm/yyyy HH:MM\'

\%

\% WARNING: as file access occurs in the background (i.e. calls to
\"sz_readTimeStep\") this object it may

\% be slow to respond, particularly for a large number of files.

\% Therefore be careful accessing the time dimension with a \":\"

\% e.g. selfeobj.hvel(:,:,:,:) may be slow to respond or run out of
memory if there are a large number of files.

\% Example of using provided example files, trisurf plot of surface
elevations at time 1:

\% %load SELFE object from Example directory provided

\% dirin=which(\'selfe\');

\% \[dirin file\]=fileparts(dirin);

\% cd(dirin)

\% slf=selfe(\'..\\Example\\\');

\% plot elevations figure;

\% trisurf(slf.elem,slf.x,slf.y,slf.elev(1,:)\');

\% view(\[0 90\]);

\% shading interp;
