### Save Project As

Allows you to save the project with another name.

![](Mimics Reference Guide/../Resources/Images/SaveAs_498x447.png)

#### Save as Type

Mimics Project Files | Saves the project in the DICOM patient coordinate system. By default the file will be saved as the current version of Mimics file format and it can contain multiple images.  
---|---  
Mimics 18.0-20.0 Project files | Saves the project in the DICOM patient coordinate system in Mimics 18.0-20.0 file format. In case the file contains more than one images at the time of saving in this file format, only the currently active images and the objects linked to it will be saved.   
Mimics 11 Project file | Saves the project in the STL coordinate system. The STL coordinate system does not correctly handle oblique images. If an oblique image set is saved as Mimics 11 it will lead to differences in sizes of 3D objects. This file format does not support 16-bit images.   
Mimics 12-13 Project file | Saves the project in the STL coordinate system. The STL coordinate system does not correctly handle oblique images. If an oblique image set is saved as Mimics 11 it will lead to differences in sizes of 3D objects. This file format does not support 16-bit images.   
Mimics 14-15 Project File | Mimics 14-15 Project File Save the project according to the old DICOM patient coordinate system. The images will be shifted with half a slice increment.  
Mimics 16-17 Project File | Saves the project in the DICOM patient coordinate system.  
  
**Mimics 18.0 file format**

In Mimics 18.0, the internal file format has been changed to facilitate creation of projects from large datasets. There is no longer a file size limitation as opposed to the 4GB limitation in previous versions of Mimics. Projects created in Mimics 18.0 will be saved in the new format. Projects created in earlier versions are still saved as the previous file format (Mimics Project Files format) by default.

#### Save images compressed

When saving the project, you can choose to compress it. The compression algorithm used is a lossy JPEG compression. Using this option will reduce the size of the project.

You can choose between three different quality presets for the JPEG compression:

When saving a project, it is possible to compress the images in the project with a lossy JPEG compression. This way you can reduce the size of your projects. It is possible to choose between three quality presets:

  * High Quality � Low Compression


  * Medium Quality � Medium compression


  * Low Quality � High compression


The higher the compression factor is, the smaller your files will become, but the worse your image quality will be.

#### Keep reduction

When you have done a reduction on the images during loading or importing, you can also choose to keep the reduction when you are saving the project.
