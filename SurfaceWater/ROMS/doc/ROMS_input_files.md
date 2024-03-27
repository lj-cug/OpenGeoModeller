# Input Parameters Files

https://www.myroms.org/wiki/Input_Parameter_Files

## 输入文件模板

ROMS has several ASCII text files to set application input parameters. 
The templates for these files are located in the User/External or ROMS/External subdirectories:
```
    Physical input parameters (default standard input file): roms.in.
    Multiple models coupling input parameters: coupling.in.
    Biological models input parameters: bio_Fennel.in, ecosim.in, nemuro.in, npzd_Franks.in, npzd_iron.in, npzd_Powell.in.
    Sediment model input parameters: sediment.in.
    Stations input parameters: stations.in.
    Floats input parameters: floats.in.
    4DVar data assimilation input parameters: s4dvar.in.
    SWAN wave model input parameters: swan.in.
```

## Important Notes on File Syntax

    Before revision 460 filename length was limited to 80 characters. Now the limit is 256 characters.

    Comment lines are allowed and begin with an exclamation mark (!) in column one. Comments may appear to the right of a parameter specification to improve the documentation of your choices. All comments will be ignored during reading.

    Blank lines are also allowed and ignored.

    Continuation lines in a parameter specification are allowed and must be preceded by a backslash ( \ ).

    Multiple NetCDF files are allowed for input field(s). This is useful when splitting input data (climatology, boundary, forcing) time records into several files (say monthly, annual, etc). In this case, each multiple file entry line needs to be ended by the vertical bar ( | ) symbol. For example:

    ! Input forcing NetCDF file name(s).

         NFFILES == 6  6                                          ! number of forcing files for 2 nested grids

         FRCNAME == my_lwrad_year1_grid1.nc |
                    my_lwrad_year2_grid1.nc \
                    my_swrad_year1_grid1.nc |
                    my_swrad_year2_grid1.nc \
                    my_winds_year1_grid1.nc |
                    my_winds_year2_grid1.nc \
                    my_Pair_year1_grid1.nc  |
                    my_Pair_year2_grid1.nc  \
                    my_Qair_year1_grid1.nc  |
                    my_Qair_year2_grid1.nc  \
                    my_Tair_year1_grid1.nc  |
                    my_Tair_year2_grid1.nc  \

                    my_lwrad_year1_grid2.nc |
                    my_lwrad_year2_grid2.nc \
                    my_swrad_year1_grid2.nc |
                    my_swrad_year2_grid2.nc \
                    my_winds_year1_grid2.nc |
                    my_winds_year2_grid2.nc \
                    my_Pair_year1_grid2.nc  |
                    my_Pair_year2_grid2.nc  \
                    my_Qair_year1_grid2.nc  |
                    my_Qair_year2_grid2.nc  \
                    my_Tair_year1_grid2.nc  |
                    my_Tair_year2_grid2.nc
    Notice that NFFILES is 6 and not 12 for each grid or 24 total. There are 6 uniquely different colored fields in the file list for each grid, we do not count file entries followed by the vertical bar symbol. This is because multiple file entries are processed in ROMS with derived type structures. In nested grids, there is not need to specified the forcing fields for all grids since ROMS will populate the same filenames for all nested grids. The fields for grid1 (coarser grid) are used to interpolate the forcing for grid2 and so on. If the horizontal number of grid points for a forcing field is different than the grid domain dimensions, it will automatically trigger reggriding inside ROMS after the forcing data is read. 

    Input parameters can be entered in any order, provided that the parameter KEYWORD (usually, uppercase) is typed correctly followed by = or == symbols for singular and plural assignments, respectively. A singular assignment indicates that there is not nested grid dependency whereas the plural assignment indicates that each nested grid has a parameter value. 

    In multiple levels of nesting and/or several connected domains, Ngrids entries are expected for some of these parameters. The double equals symbols == syntax is for those parameters which will need to be assigned distinct values for each grid in multiple grid applications. The order of the entries for multigrid parameters is essential. It must follow the same order (1:Ngrids) as in the state variable declaration.

    Parameters that are not used can be omitted from the list, so the input file for a specific application can be very concise.

    Frequently, more than one value is required for a parameter. If fewer values are provided, the last value is assigned for the entire parameter array. For convenience, the multiplication symbol (*) without blank spaces in between is allowed for a repeated value in a long list specification. For example, in a three grids nested application AKT_BAK must be specified for temperature and salinity for all three grids, i.e. six values in total. The line:
         AKT_BAK == 2*1.0d-6  2*5.0d-6  2*3.0d-6                     ! m2/s
    indicates that the first two entries of array AKT_BAK will have the same value of 1.0d-6 for grid 1, the next two entries will have the same value of 5.0d-6 for grid 2, and grid 3 will use 3.0d-6. Thus the line is short-hand for:
         AKT_BAK == 1.0d-6 1.0d-6 5.0d-6 5.0d-6 3.0d-6 3.0d-6        ! m2/s
    The comment at the end is provided as a reminder of the correct units for this parameter.
