# MapPy

MapPy is a Python tool to extract elevation contour lines from raster files and save them as shapefile.
You need to have a copy of the folder customtkinter of CustomTkinter in your working folder.
You can download it from the [CustomTkinter repository](https://github.com/TomSchimansky/CustomTkinter).

MapPy can read a svg file or a jpeg file. To use a jpeg file you need to install [AutoTrace](https://github.com/autotrace/autotrace/releases)
and add the path in MapPy (line 1068).

If you do not have autotrace installed, you can create the .svg file with Inkscape, usinc the function Trace Bitmap with Detection Mode set to "cenerline tracing".


It is also pissible to download a Docker container with MapPy and all the required tools:

docker pull demichie/mappy

To launch the container you have to execute the script run.sh you can find in this repository. Put it in the folder where you have your maps in .jpg format, and make the file executable with :

chmod +x run.sh

On Mac computer you also need xquartz:

https://www.xquartz.org/

To use MapPy with xquartz, please follow steps 2-6 from this guide:

https://gist.github.com/sorny/969fe55d85c9b0035b0109a31cbcb088

