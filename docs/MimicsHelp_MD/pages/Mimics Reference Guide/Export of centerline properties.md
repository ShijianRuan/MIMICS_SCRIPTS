Export Centerline Properties

The centerline export dialog can be reached via the centerline properties dialog, and after selecting the branch sets which you want to be exported.

Hint: the centerline can also be used in Matlab via the Matlab function (see the section on 'Run Matlab Script').

![](Mimics Reference Guide/../Resources/Images/image17.png)

Export  
---  
Output Directory |  Select the destination folder.  
File Name |  Select the file name  
Save as type |  The centerlines can be exported as Text or as Iges file.  The text file will list the position and measurement information of the control points. The information is grouped per branch segment (i.e. a part of the centerline between two branching points or end points)and is listed in columns. The coordinates of the control points are indicated by Px, Py and Pz, the coordinates of the tangent vector in the control point are represented as Tx, Ty, and Tz, the ones of the normal vector are represented as Nx, Ny and Nz, and the ones of the binormal vector as BNx,BNy, and BNz.  The Iges file exports the control points and the connection between them. Besides the control points also the best fitted, minimal and maximal diameter can be exported as an Iges file.  
Measurements  
Best fitted diameter |  Diameter (mm) of the best fit circle in each control point. The measurement is exported as Dfit.  
Minimal diameter |  Diameter (mm) of the inscribing circle in each control point. The measurement is exported as Dmin.  
Maximal diameter |  Diameter (mm) of the subscribing circle in each control point. The measurement is exported as Dmax.  
Curvature |  Curvature at each control point (definition: see below). The measurement is exported as C.   
Hydraulic Diameter |  Hydraulic diameter (mm) at each control point. The Hydraulic diameter is exported as Dh.  
Hydraulic Ratio |  Hydraulic ratio at each control point. The Hydraulic ratio is exported as Xh.  
Circumference |  Perimeter (mm) of the cross section normal to the centerline at each control point. The measurement is exported as Scf.  
Sectional area | Area (mm�) of the cross section normal to the centerline at each control point. The measurement is exported as Area.  
Ellipticity |  Ellipticity of the best fit ellipse at each control point (definition: see below). The measurement is exported as E.
