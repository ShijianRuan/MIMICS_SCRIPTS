# S-curve

The S-curve or the optimal projection curve is an important parameter which is required in the planning of cardiovascular procedures like Transcatheter Aortic Valve Implantation (TAVI) or Transcatheter Mitral Valve Replacement (TMVR). It is needed to achieve accurate device positioning during the interventions. The S-curve represents all the possible angles where the native valve plane can be seen edge on (=visible as a line).

In practice, the curve is made by plotting the values of the Cranial & Caudal angles for step changes in the LAO & RAO angles. The left and right oblique angles are changed at a fixed interval and the respective cranial and caudal angles are noted.

The S-curve can be calculated automatically for a defined plane. The range of the S-curve can be set manually and the S-curve will be generated for the defined range of angles.

**S-curve Plane**

The S-curve Plane shows a list of Analysis, simulation, mirror and reslice planes which can be used as input for creating the S-curve.

![](Mimics Reference Guide/../Resources/Images/S-Curve_planes.png)

Note: If the analysis and/or simulation module have not been activated, it will not be possible to use analysis, simulation or mirror planes for the generation of the S-curve. It is however possible to use the reslice plane as an input for the S-curve.

**S-curve Viewport**

![](Mimics Reference Guide/../Resources/Images/S-Curve.PNG)

The S-curve is displayed in the graph viewport (right). Here, you can define the range and step increments of the S-curve based on your preference.

The display area shows the S-curve in red with LAO/RAO values on the horizontal axis and CRAN/CAUD values on the vertical axis. The images and the 3D viewport are automatically updated when a control point on the S-curve is selected on the graph viewport. The control points of the S-curve can also be sequentially browsed by using the ![](Mimics Reference Guide/../Resources/Images/AngleIncrementButton.png) ![](Mimics Reference Guide/../Resources/Images/AngleIncrementButton02.png) buttons at the bottom of the graph viewport.

RAO-LAO Rotation Range |  Defines the range of the S-curve.  
---|---  
Step increment | Defines the interval between two points representing the RAO/LAO angles in the S-curve.
