#### Reading Tiff, Bitmap and Jpeg images

When you select a set of images in Bitmap, Tiff or Jpeg format an extra window will be displayed on the screen after clicking Next in the first Import images new project wizard window. This is the Images Property window. In this Bmp/Tiff/Jpeg Import window you can fill in some image related parameters and order your images before the conversion is performed. The radiologist usually provides this information.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000078_722x476.png)

Filenames

The filenames of the image data stack that you wish to import can be seen in on the left. When you select an image, the up-down arrow buttons become active. These allow you to position the images in a custom order. You can view the images in the preview as well. 

Sorting Order

You can select how the images should be sorted. The available options are numerical ascending, numerical descending, alphabetical ascending or alphabetical descending orders. The sort order can be changed interactively by selecting an image in the list and clicking the Up or Down arrow in the Move box. The image will be moved up or down one position in the list.

Scan resolution

In the scan resolution section, it is possible to give the resolution of the scan along the x, y and z directions. It also is possible to indicate in what dimension this resolution is measured as well from the drop down menu. 

When isotropic sampling is checked, the values in x, y and z will be the same as the x value. 

Study information

Here you can enter the Patient Name and Institute Name that is relevant to your project.

##### Edit images

The edit images dialog allows you to crop your images, resample your images and change the pixel mapping of the images. 

###### d. Volume crop/resize

Here you can crop and resample your images. There are two ways to crop your image: 

  1. Enter values in the crop dialog box, or
  2. Move the bounding box (shown in white) on the 2D image views. 


To resample your data, you can: 

  1. Scale the data,
  2. Change the pixel size, or 
  3. Skip some images. 


Resampling is especially useful if you have a large dataset. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000079_605x502.png)

###### e. Pixel mapping

Here you can map the original grayscale value to a custom range. This allows clubbing together all pixels above and below your original range to one value and gives you more values in the regions of interest. This makes it easier to differentiate in between different materials.

It is also possible map the range of the input grayscale images from 16 bit to 8 bit grayscale range. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300007A_605x502.png)

**Unit selection (μm-mm)**

The scan resolution of the project can be manually entered with a precision of 4 decimals in the semi-automatic and manual import. If a pixel size less than 0.01 is entered in the scan resolution, the following warning message will appear in the dialog box.

![](Mimics Reference Guide/../Resources/Images/um-mm_624x400.png)

If a project is created with a pixel resolution of less than 0.01, some measurements might be rounded. Therefore it is advised to select the optimal unit during import.

A project will be created with the unit defined during the import of images and will retain this unit after creation. To create a project with a different unit, you can import the images and select the desired unit. 3D models, CAD objects and masks will be appropriately rescaled when copied from one Mimics project to another project with different units.

**Note** : It is not possible to add a non DICOM group of images if there is already an existing group of non DICOM images in the current import wizard session. 
