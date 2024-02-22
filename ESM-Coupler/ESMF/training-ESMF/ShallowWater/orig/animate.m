% Octave script
%
% Creates a series of surface plots of h
% over the 2D domain.  A program like mencoder
% can then be used to stitch together an
% animation.
%
% mencoder command used on Ubuntu 12.04:
% mencoder mf://*.png -mf w=640:h=480:fps=25:type=png -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy -o swout.avi
%
% Author: Rocky Dunlap   <rocky.dunlap@noaa.gov>

pkg load netcdf;

% NetCDF file should be in same directory
% Modify filename below if required

info = ncinfo("swout.nc");
noutput = info.Dimensions(3).Length;
x = ncread("swout.nc", "x");
y = ncread("swout.nc", "y");
h = ncread("swout.nc", "h");

%set(0, 'defaultfigurevisible', 'on');
 
for t=[53:100]
    fig = figure();
   
    %for some reason, I am having to make the plots visible
    %on screen for print() to save them to file
    %this slows things down considerably...
    %set(fig, "visible", "off");
    
    surf(x,y,h(:,:,t));
    axis([0 30000000 0 5000000 9500 10800]);
    shading interp;
    xlabel("x");
    ylabel("y");
    zlabel("h");
    filename=sprintf("output/%04d.png", t);
    disp(["Filename = " filename]);
    print(fig, filename, "-dpng");
end


