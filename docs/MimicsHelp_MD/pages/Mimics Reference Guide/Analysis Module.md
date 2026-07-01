### Analyze

Gain an in-depth understanding of anatomy and pathology as well as device fit and performance using advanced quantification tools.

  * Indicate landmarks and annotate 
  * Characterize anatomical features with primitives and centerlines
  * Measure distances and angles in 3D
  * Compare aligned objects in a click
  * Analyze curvature or thickness
  * Use integrated Matlab� scripting for complex analysis


Analysis module is designed to make a bridge between medical imaging (CT and MR) and CAD design. This means that it can export data from the imaging system to the CAD system and vice versa.

Use the right tools in the right software.

Analysis module is not intended for design. Many software packages are available to make designs and most designers are already familiar with a certain CAD package. Learning yet another CAD system would be a waste of time.

A CAD software is not an imaging software. If you try to convert all the medical data to your CAD system, your system will slow down drastically. Not all the data from the images can be converted to the CAD system: only a surface representation of the images is visible in the CAD system.

Therefore it is better to make the interpretations on the images with Analysis objects, taking into account all the gray value information (soft tissue, different types of bone, tendons, etc.�). On the images, basic features can be recognized and converted to geometrical entities (basic axes, reference curves, anatomical landmarks). Only the surfaces which are needed to make a fit are really converted to B-spline surfaces.

On the other hand, CAD design can be imported in the Mimics via the STL interface. The designs are visualized in 2D sections together with the actual images, or in 3D shaded representations, with the anatomical data in a transparent mode.

With this method, it is possible to bridge between the images and the CAD system in a few minutes time.

All Analysis functions are loaded immediately in Mimics after registration of the Analysis module. There is no specific way to start Analysis module. When the module is registered the extra feature appears in the interface.

On the menu bar:

  * The Analyze menu lists geometrical objects that can be drawn freely or fitted onto polylines. It also allows you to fit splines and surfaces on polylines.


In the File->Export menu:

  * Two new export file formats are added: Point Cloud and Iges.


![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030001A7.png)

In the Project Management:

  * Objects tab page.


  * Extra options are added to the action list of the Polylines tab: select a polyline-set and click on the action button to fit an Analysis object on the polylines or to export the polylines as Iges.


![](Mimics Reference Guide/../Resources/Images/PMTab_AnalysisObjects.PNG) |  ![](Mimics Reference Guide/../Resources/Images/PMTab_Polylines.PNG)  
---|---  
  
Note: If you're not able to start Analyze module, it is possible that the correct password has not been used. Go to Help > Modules and fill in the correct passwords.
