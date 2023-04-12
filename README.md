# MapPy

MapPy is a Python tool to extract elevation contour lines from raster files and save them as shapefile.
You need to have a copy of the folder customtkinter of CustomTkinter in your working folder.
You can download it from the [CustomTkinter repository](https://github.com/TomSchimansky/CustomTkinter).

MapPy can read a svg file or a jpeg file. To use a jpeg file you need to install [AutoTrace](https://github.com/autotrace/autotrace)
and add the path in MapPy (line 1068).

If you do not have autotrace installed, you can create the .svg file with Inkscape, usinc the function Trace Bitmap with Detection Mode set to "cenerline tracing".

