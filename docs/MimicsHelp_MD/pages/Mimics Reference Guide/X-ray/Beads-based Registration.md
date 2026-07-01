# Beads-based Registration

![](Mimics Reference Guide/X-ray/../../Resources/Images/Beads-based%20Registration.png)

The X-ray module provides an X-ray object registration method that uses the position of beads in the 3D image data and the X-ray images as an input. First, the beads should be indicated on the X-ray images and in the 3D image dataset. Next, the beads are paired so the beads in the 3D image data are linked to the matching beads on the X-ray image. When the beads are paired, a registration algorithm can accurately position the selected X-ray objects in the 3D space. The last step allows to create Analysis spheres from the registration result.

Required input:

  * 3D image data including beads
    * Minimum 4 beads are required to perform the workflow.
  * 2 X-ray objects
    * Best results are obtained if the 2 X-rays have a relative angle of approximately 90� to each other.


1\. **Select X-ray Objects**

Select two different X-ray objects, one in each list.

![](Mimics Reference Guide/X-ray/../../Resources/Images/BeadsBasedRegistration_SelectXrayObjects_618x283.png)

2\. **Place beads**

When proceeding to the Place beads step, the window layout will change and combine the standard image layout and the 2D X-ray viewports of the selected X-ray objects.

![](Mimics Reference Guide/X-ray/../../Resources/Images/BeadsBasedRegistration_Layout_620x232.png)

**Placing 2D beads**

In the 2D X-ray viewports, a single click inside the white dot of the bead will automatically fit a circle. If the bead gradient is not detected, a circle with default diameter will be created. The circle is solely used as a visual guide to match the circular bead. Make sure the center point matches the center of the bead. Move the entity by simply click and drag the center point of the circle. The diameter of the circle can be adjusted by clicking and dragging any point on the circle. Delete a bead by clicking the center point and pressing delete or via the context menu. The placed beads will automatically be stored as a 2D Beads set of the respective X-ray Object in the PM tab.

![](Mimics Reference Guide/X-ray/../../Resources/Images/BeadsBasedRegistration_2DBeadsPlacement_622x187.png)

**Placing 3D beads**

Beads in the 3D image dataset are created with a single click in one of the 2D image viewports. By default a spherical bead will be created of which the contour will be visible in the image viewports. The placed beads will automatically be grouped in the 3D beads set in the PM tab. The diameter of the sphere can be changed by clicking and dragging any point of the contour. To move the bead, scroll to the slice with the center point of the sphere and drag the point. Delete a 3D bead by selecting the center point and pressing delete or via the context menu.

![](Mimics Reference Guide/X-ray/../../Resources/Images/3DBeadsBasedRegistration_Placing3DBeads_584x94.png)

Pair Beads

![](Mimics Reference Guide/X-ray/../../Resources/Images/PairBeads_1.png) |   
---|---  
  
When clicking on a numbered bead, a window will pop-up which allows you to assign a particular number, unassign the selected bead, or unassign all the beads numbering. Once all the beads are numbered, rotate the 3D viewport so the 3D beads align with the beads on the X-ray.

![](Mimics Reference Guide/X-ray/../../Resources/Images/PairBeads_3.png)

**Perform Registration**

To perform the X-ray object registration, at least 4 bead pairs should be indicated. After pressing Perform registration the selected X-ray objects will adjust their position. Now, the bead projection lines (the line from a bead on the X-ray to the X-ray source) will cross each other at the paired bead in the 3D image data.

![](Mimics Reference Guide/X-ray/../../Resources/Images/BeadsBasedRegistrationo_PerformRegistration.png)

**Create 3D beads**

Pressing �Create 3D beads� will create new beads (Parts in the PM tab) based on the registration result. The contour projection of the Parts can be used to visually check the accuracy of the registration.
