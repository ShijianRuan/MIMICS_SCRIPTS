# X-ray Objects

The X-ray tab has a tree structure that will contain the following new entities:

  * X-ray Object
    * Contour
    * Beads Set
    * Point Set
      * Point
  * 3D Beads Set


![](Mimics Reference Guide/X-ray/../../Resources/Images/X-rayTab-01.png)

**List of created Objects**

Name | Name of the object. By clicking on the name of the object, it can be renamed.  
---|---  
Visible | Lists if the object is visible or not by means of glasses. You can change the visibility by clicking on the glasses.  
2D |  The 2D visualization of an X-ray Object in the X-ray layout can be toggled on by ticking the checkbox. If it is not checked, the X-ray Object will not appear in a 2D viewport of the X-ray layout.   
  
**Functions on Objects**

Import X-ray | This tool allows you to import 2D images and will convert them into X-ray Objects.  
---|---  
Delete | Deletes the selected object.  
Properties | Displays the properties of the selected object.  
Duplicate | Creates a duplicate of the selected object. (Only enabled for X-ray Objects)   
  
# X-ray Object Properties

An X-ray object represents the imported X-ray in the 3D space. It is visualized as a pyramid object with the X-ray image at the base and the X-ray source at the top. The color of the pyramid lines can be changed by ticking the colored label which will open the color picking window. To hide the construction lines of the pyramid, uncheck the checkbox.

![](Mimics Reference Guide/X-ray/../../Resources/Images/X-rayObjectproperties.png)

**Projected Contours**

It is possible to change the contour projection from the �projected Contours� tab. Select one of the radio buttons to hide (�Off�), show only outer contours (�Outer�) or show all contours (�All�) of a particular Part from the list.

**DICOM tags**

All the DICOM tags of the imported X-ray are organized and listed in the �DICOM tags� tab.

**Detector Position**

The position of the X-ray object is determined by the location of the �Center� (center of the X-ray image) and the direction of the �Direction to Source� (from the X-ray image center to the X-ray source) and �Plane Direction� (from the X-ray image center to the top of the X-ray image).

**Source Position**

The �Distance Source to Detector� can be edited in the X-ray object properties dialog box, regardless of the DICOM tag that is read during the import.

It is possible to edit the angle of the X-ray object in case the X-ray image is acquired with an angle. The X-ray source can be rotated in direction of the top or bottom of the image. The defined �Distance Source to Detector� Value will always be kept.

![](Mimics Reference Guide/X-ray/../../Resources/Images/X-rayObjectproperties-2_608x341.png)
