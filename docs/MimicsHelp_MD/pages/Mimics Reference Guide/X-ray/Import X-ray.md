# Import X-ray  
  
The X-ray module allows you to import X-rays ![](Mimics Reference Guide/X-ray/../../Resources/Images/Import%20X-ray.png) into your Mimics project. Mimics supports importing of the following image formats:

  * DCM - DICOM images
  * JPEG
  * BMP
  * TIFF
  * Raw Images


The imported X-ray images will be converted into X-ray Objects in your current Mimics project. These X-ray Objects will appear as pyramid like objects having the image at the base and the X-ray source at the top. The size of the image will be determined by a set of image properties.

![](Mimics Reference Guide/X-ray/../../Resources/Images/X-ray_Acquisition_001.png)

Note: X-rays cannot be imported through the new project wizard. If the new project wizard is run, the images will be used to create a new project instead of being converted into an X-ray object.

_Required X-ray image properties_

  * Pixel width and height: the amount of pixel rows and columns that compose the image.
  * Pixel spacing: the distance between the centers of neighboring pixels.
  * Distance source to detector: the distance between the center of the image and the X-ray beam source.


When launching Import X-ray, the first step is to select one or several images in the file browser window to be imported. Use CTRL-click to individually select multiple images or SHIFT-click to select a list.

![](Mimics Reference Guide/X-ray/../../Resources/Images/ImportXray_001.png)

The next page of the wizard step will be customized depending on the selected image format. If the image format is not DICOM, a set of required image properties need to be filled in manually.

In order to correctly import your DICOM X-ray images, the following acquisition properties need to be known:

DICOM tag | DICOM tag name |  Description  
---|---|---  
(0018,1110) | Distance Source to Detector | Distance in mm from source to detector.   
(0018,1164) | Imager Pixel Spacing  | Physical distance measured at the front plane of the Image Receptor housing between the centers of each pixel. Specified by a numeric pair - row spacing value (delimiter) column spacing value - in mm. In the case of CR, the front plane is defined to be the external surface of the CR plate closest to the patient and radiation source.  
(0028,0010) | Rows | Amount of pixels in height, Y-direction  
(0028,0011) | Columns | Amount of pixels in width, X-direction  
(0008,0060) | Modality | The image modality tag should always be �CR� (Computed Radiography).  
  
After pressing **Next** , the selected images will be shown as a list of studies. When clicking on one of the images from the list, a preview of the image will be generated. If a DICOM image is selected, the attached DICOM tags will be organized in the tabs.

![](Mimics Reference Guide/X-ray/../../Resources/Images/X-ray/xray_import.png)

All images that are checked will be converted and imported in the Mimics project where they will appear in the X-ray Object PM tab as �**X-ray Objects** �.

You can create custom tags tab, clicking the ![](Mimics Reference Guide/X-ray/../../Resources/Images/dicom_tags_add_tag_button.png) button. In the pop window you can set the name of the custom tags tab.

To add a tag in the custom tags tab, right click to the selected tag and select **Add Tag to Tab**.

![](Mimics Reference Guide/X-ray/../../Resources/Images/dicom_tags_add_tag_to_tab_277x79.png)

_Raw Import_

When importing Raw images, a set of image properties need to be manually entered. The Scan resolution, Distance Source to Detector, Image parameters, and the Pixel properties are required to successfully import and convert the images. Adding Study Information is optional. The options for pixel properties can be chosen from one the drop down menus.

If the required fields (yellow) are filled in, the Convert button will become enabled. When clicking Convert, the Raw image will be converted into an X-ray Object and imported into the current Mimics project.

![](Mimics Reference Guide/X-ray/../../Resources/Images/RawImport.png)

_Import .JPEG, .BMP or .TIFF image_

The X-ray module supports importing images which have the .JPEG, .BMP or .TIFF format. When selecting images in one of the mentioned formats, you will be requested to fill in the required image properties (yellow fields). Once the required fields are filled in, the images can be converted to X-ray Objects.

Note: �Import X-ray� only supports images with square pixels so the pixel size in X and Y direction is always kept equal.

![](Mimics Reference Guide/X-ray/../../Resources/Images/2014-03-30_19h24_33.png)
