##### Centerline Measurements

A complete range of measurements can be made on the calculated centerline by clicking on _New_ in the Measurements project management tab, by choosing the Measure toolbar, or **Measure > Centerline** in the menu.

![](Mimics Reference Guide/../Resources/Images/Centerline_Menu.png)

Best Fit Diameter |  Diameter of the best fit circle in a control point. The measurement is exported as Dfit.  
---|---  
Circumference | Perimeter of the surface in a control point. The measurement is exported as Scf.  
Sectional Area | Area of the sectional surface normal to the centerline. The measurement is exported as Area.  
Minimal Diameter |  Diameter of the inscribing circle in a control point. The measurement is exported as Dmin.  
Maximal Diameter |  Diameter of the subscribing circle in a control point. The measurement is exported as Dmax.  
Curvature |  Curvature of the centerline in the indicated point  
Tortuosity |  Tortuosity of the segment between the two points indicated in the measurement interface. Tortuosity = 1 - (linear distance / distance along the branch)  
Hydraulic Diameter |  Hydraulic diameter in the indicated point. The hydraulic diameter is exported as Dh. Hydraulic diameter = 4 * Sectional Area / Circumference  
Hydraulic Ratio |  Hydraulic ratio in the indicated point. The hydraulic ratio is exported as Xh. Hydraulic ratio = Hydraulic diameter / Maximal diameter  
Distance Over Centerline |  Shortest distance between points P1 and P2 over centerline.  
Triad | TNB (Frenet-Serret) frame visualizing, and measuring the tangent, normal and binormal vector directions in a point selected on the centerline. Color code for the tangent, normal, and binormal is yellow, green, and blue respectively. To create a triad measurement use left mouse click to pin the location and at the same point use right mouse click to create the measurement.  
Ellipticity | Ellipticity of the best fit ellipse at each control point. The measurement is exported as E. Ellipticity ![](Mimics Reference Guide/../Resources/Images/Ellipticity%20formula_75x37.jpg) with R1 the largest radius and R2 the smallest radius.
