##### Using Mimics with AnyBody Modeling System

In order to create subject-specific musculoskeletal models, Scaling function in AnyBody Modeling System (AMS) morphs generic to subject landmark points and bones in a step-wise procedure, and updates the location of muscle and ligament attachment points accordingly. First step of Scaling function is to use bony landmarks selected on subject and generic bones. In collaboration with AMS, a template including a set of bony landmarks has been defined for the femur bone. These bony landmarks should be selected on subject femur bone and exported to AMS, while a similar template is provided for generic femur bone in AMS. 

The AMS femur template includes the following points:

  1. **Medial Head Point:** A point located ca. 5mm posteriorly from fovea capitis
  2. **Anterior Lateral Condyle Poin** t: The outermost point on the anterior side of the lateral condyle surface
  3. **Anterior Medial Condyle Point:** The outermost point on the anterior side of the medial condyle surface
  4. **Posterior Greater Trochanter Point:** Superior peak point of the intertrochanteric crest
  5. **Anterior Greater Trochanter Point:** Anteriorly and laterally positioned point of the greater trochanter, please refer to the visual aid
  6. **Anterior Shaft Point:** Anteriorly positioned mid-femur point
  7. **Posterior Lateral Condyle Point: T** he outermost point on the posterior side of the lateral condyle surface
  8. **Posterior Medial Condyle Point:** The outermost point on the posterior side of the medial condyle surface
  9. **Posterior Head Point:** The outermost posterior point on the femoral head, coincides with the femoral head center if projected onto the coronal plane
  10. **Anterior Head Point:** The outermost anterior point on the femoral head, coincides with the femoral head center if projected onto the coronal plane
  11. **Proximal Head Point:** The most superior point on the femoral head, coincides with the femoral head center if projected onto the transverse plane
  12. **Proximal Greater Trochanter Point:** Anteriorly and proximally positioned point of the greater trochanter, please refer to the visual aid
  13. **Medial Lesser Trochanter Point:** Most Medial point on lesser trochanter
  14. **Distal Trochanteric Fossa Point:** Most distal point of the trochanteric fossa
  15. **Posterior Trochanteric Fossa Point:** Most posterior point of the trochanteric fossa
  16. **Proximal Posterior Greater Trochanter Point:** Most posterior-proximal peak point of the intertrochanteric crest 
  17. **Lateral Distal Trochanteric Fossa Point:** Point on the lateral side of the greater trochanter at the same proximo-distal level as the distal trochanteric fossa point
  18. **Lateral Lesser Trochanter Point:** Point on the lateral side on the femur shaft at the same proximo-distal level as the medial lesser trochanter point
  19. **Femoral COR:** Center of rotation of femoral head


![](Mimics Reference Guide/../Resources/Images/MSM%20tutotial/03000001_517x564.png) ![](Mimics Reference Guide/../Resources/Images/MSM%20tutotial/03000002_545x448.png) ![](Mimics Reference Guide/../Resources/Images/MSM%20tutotial/03000003_607x499.png) ![](Mimics Reference Guide/../Resources/Images/MSM%20tutotial/03000004_187x469.png)

![](Mimics Reference Guide/../Resources/Images/MSM%20tutotial/03000005_470x272.png)

![](Mimics Reference Guide/../Resources/Images/MSM%20tutotial/03000006_576x268.png)

These points are used in a first step of the AMS Scaling function. Therefore, inaccurate location of (some) the points will not jeopardize the results of the AMS Scaling. In case some of the points cannot be selected due to low image resolution, or limited image field of view, points should be removed from the subject template, as explained later (�Editing template�), as well as from the generic template in AMS. 

###### Getting Started

The pre-defined AnyBody femur template tool allows for selecting a set of femur bony landmarks to be used in AnyBody to morph generic femur to subject femur. To use the AnyBody femur template, go to the Simulation menu and choose Measure and Analysis.

###### Choosing the type of analysis

Before starting the selection of femur bony landmarks, you first have to choose the �**AnyBody femur landmarks** � template in the pull down menu.

![](Mimics Reference Guide/../Resources/Images/MSM%20tutotial/03000007_221x564.png)

###### Points of analysis

The points of the analysis pane provide you with a list of pre-defined points in the currently selected template. The pane allows you to indicate, locate, edit or clear the points in the list. 

###### Indicating points

To place a point, first select it from the list and click on the �Indicate� button. You can indicate a point in both the 2D and the 3D views. Note that you first need to indicate the point before you can use any of the other option in the pane. When you have clicked the Indicate button, the description of the point will be displayed.

![](Mimics Reference Guide/../Resources/Images/MSM%20tutotial/03000008.png)

![](Mimics Reference Guide/../Resources/Images/MSM%20tutotial/03000009.png)

You can always move the points of the analysis after their indication. If you are not able to select the landmark points, first enable the right mouse mode by clicking on the Indicate button. Now you will be able to move the points.

###### Locating points

If you want to easily view the image on the location where a point was placed, highlight the point and click on �locate�. This will move both the axial and the sagittal view to the position where the point is located. 

###### Clearing points

If you have misplaced a point you can easily remove it by selecting the point in the list and clicking on the �delete� button.

###### Editing points

After having indicated a point you can change its properties. If you click on the �edit� button you can easily change the color in which the point is shown on the images. You can also change the position of the point here by changing its coordinates.

###### Exporting points

When the points are correctly located, they can be exported as xml file that is compatible with AMS.

![](Mimics Reference Guide/../Resources/Images/MSM%20tutotial/0300000A_218x218.png)

###### Editing template

In case you do not want to include some of the points defined in the template, do not indicate these points, and if they have been already located, delete them. The name of these points appears in grey in the selection pane. When exporting points, these points will not appear in the xml file.

In case you want to add new points in the template, you need to create a new template. Go to Overview, Select AnyBody femur template, and click on Change. A copy of the template will automatically be created that you can rename. To add some points, click on New in Points pane, and create as many points as needed. Once these points are created, same properties (Indicate, Locate, Edit, Delete) than for pre-defined points apply to them.
