### DICOM  
  
This function allows converting the imported images in DICOM format. You can also add the information of the segmentation masks or the contours of Parts, providing you with a straightforward link to virtual surgery navigation systems. You just need to indicate the objects you want to superimpose on the images and browse to a directory where you want the DICOM files to be saved. Possible objects are masks, the contours of 3D and Analysis objects and STL files and Simulation objects. You can also specify the thickness of the contour in the DICOM images.

![](Mimics Reference Guide/../Resources/Images/Export_DICOM.png)

_Note_ : The tool does not support the export of DICOM Ultrasound images and scout images.

_Note:_ Depending on the amount of information present in the DICOM tags at the moment of the export, the exported images will either be written as CT/MRI DICOM images or as Secondary Capture DICOM Images.
