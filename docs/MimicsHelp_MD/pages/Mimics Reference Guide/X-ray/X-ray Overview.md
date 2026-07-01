# X-ray  
  
The X-ray module allows you to combine a patient�s X-ray images with their CT or MRI scan data in one Mimics project. The X-ray module makes 3D measurements on X-rays possible by manual or automatic 2D/3D registration. These measurements can indicate the change in relative position of bones and implants in different conditions or points in time. This allows you to compare a post-operative implant position to the 3D surgical plan or to determine the position of bones under various loading conditions.

![](Mimics Reference Guide/X-ray/../../Resources/Images/X-ray_Menu.png)

Starting X-ray

All X-ray functions are loaded immediately in Mimics after registration of the X-ray module or package. There is no specific way to start the X-ray module. When the module is registered the extra feature appears in the interface.

On the menu bar:

The X-ray menu lists all the necessary tools to use 2D image data in your Mimics project:

  * Import X-ray deals with importing 2D images into the Mimics project.
  * The Create Contour and Show/hide Projected Contours tools support the registration methods.
  * Manual Registration allows to manually edit the position of an X-ray Object relative to a 3D model and vice versa.
  * Contour-based Registration offers a way to automatically optimize the fit of the contour projection of a 3D model with the created contour by either moving the X-ray object or the selected 3D model.
  * Beads-based Registration provides a method to automatically register 2 X-ray objects based on the indication of the beads location on both X-rays and the 3D image data.
  * Contour-based Measurement gives an indication of the accuracy of the registration by measuring the difference between the created and projected contour.
  * Calculate Part Position Difference and Measure Matching Points Distance allow making basic measurements of the X-ray registration results.
  * Create Point on X-ray Object and Reproduce Point in 3D are tools that allow the creation of a point in the 3D space based on the indication of a point on two registered X-rays.
  * Virtual X-ray tool allows creating virtual X-ray images from the patient's CT scan.


  


In the Project Management:

  * Extra X-ray Object tab is introduced with the X-ray module.
    * The new tab has a tree structure with parent-child relationships. More info about the new tab can be found in the X-ray tab help chapter.


Note: If you don�t see the X-ray menu or the X-ray Objects tab, it is possible that the correct password has not been used. Go to **Help > Modules** and fill in the correct password.
