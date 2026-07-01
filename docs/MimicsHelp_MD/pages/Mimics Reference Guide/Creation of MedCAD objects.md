### Creation of Analysis objects

On the great trochanter and on the lower part of the femur we will fit a Free Form Surface, on the femur head, we will fit a sphere.

In the Project Management on the Polylines tab, you will find a button Fit Surface. Choose Selection 2 and press the Fit Surface button. The following dialog box will appear.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002BA_323x463.png)

Surface Fit Parameters

You can accept these default values and a Free Form Surface will be fitted on Selection 2.

Note: Some caution in increasing the number of control points is advised. The basis of a B-spline is a polynomial and a polynomial has the tendency to wave. So, if the number of points is too high, the fit on the polyline will become worse.

Repeat this set on Selection 4. 

The Free Form Surfaces are visible in 3D as a shaded surface and in 2D you will see a cross-section on every layer of this Free Form Surface.

To fit a Sphere on Selection 3, go to the Analyze menu and select Sphere > Fit on Polylines. Choose the correct polyline set.

The result of all these fittings should look like following figures:

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002BB_207x187.png) |  ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002BC_207x173.png)  
---|---  
Objects fitted on the Polyline sets |  Imported STL files
