## DICOM patient coordinate update

When opening a project previously saved in MIS 15, in MIS 16 and following versions, a warning message is displayed.

![](Mimics Reference Guide/../Resources/Images/DICOM%20patient%20coordinate%20system.jpg)

You can choose to convert the project to the updated DICOM patient coordinate system, or not. The explanation on the differences can be found below.

### DICOM patient coordinate system

Mimics works in the DICOM patient coordinate system as described by NEMA in the DICOM 3.0. Required tags are:

**(0020,0032) image position (patient)** : The x, y, and z coordinates of the upper left

hand corner (center of the first voxel transmitted) of the image, in mm.

**(0020,0037) image orientation (patient)** : direction cosines of the first row and first column with respect to the patient.

The direction cosines for the orientation of the patient coordinate system determine the axis and orientation of the images. The first three values are the direction cosines of the row axis, the latter three values are the direction cosines of the columns. 

Image Orientation (Patient): 1 \ 0 \ 0 0 \ 1 \ 0

X \ Y \ Z X \ Y \ Z

Row Column

For this example the Row axis will be aligned with the X-axis of the WCS, the column axis will be aligned with Y-axis of the WCS.

The DICOM patient based coordinate system is a right handed coordinate system. The X-axis is increasing to the left hand side of the patient. The Y-axis is increasing to the posterior side and the Z-axis is increasing towards the head of the patient. 

![](Mimics Reference Guide/../Resources/Images/Coordinate%20System/03000001_554x342.png)

### Mimics 14.x and Mimics 15.x Coordinate system

Mimics 14.x and Mimics 15.x applied the DICOM patient coordinate system according to the old image position (patient) definition.

**(0020,0032) image position (patient):** The x, y, and z coordinates of the upper left

hand corner (the center of the first pixel transmitted) of the image, in mm.

In the direction of the slice, the image position will indicate the start of the slice rather than the center of the slice. This gives rise to a shift of half a slice increment. 

![](Mimics Reference Guide/../Resources/Images/Coordinate%20System/03000002_363x212.png)

### Loading 14.x and Mimics 15.x projects

On loading a Mimics 14.x or Mimics 15.x the option will be presented to update the DICOM patient coordinate system to the latest standard. 

![](Mimics Reference Guide/../Resources/Images/Coordinate%20System/03000003_427x165.png)

Note: In some cases gantry tilted project cannot be converted. You can still work on this cases in the old coordinate system. 

### Mimics 11.11 until Mimics 13.x Coordinate system 

Mimics 13.1 works in the STL coordinate system, a right handed coordinate system. The X-axis and Y-axis are respectively aligned with the images rows and the image columns. Subsequently the Z-axis is always aligned in direction of the slices.

The X-axis and Y-axis are always positioned in the upper left hand corner of the image (in the upper left corner of the first pixel transmitted). The X and Y coordinate of the upper left hand voxel�s upper left corner is always 0. 

### Loading Mimics 11.11 to Mimics 13.x projects

On loading projects made in a Mimics version between Mimics 11.11 and 13.x, the option will be presented to update the STL coordinate system to the DICOM patient coordinate system.

![](Mimics Reference Guide/../Resources/Images/Coordinate%20System/03000004.png)

Note: In some cases such as for projects with a gantry tilted, the project cannot be converted. In such case you will get a warning message, indicating the project will be in the STL coordinate system.
