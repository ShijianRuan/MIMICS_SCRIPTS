#### Images

![](Mimics Reference Guide/../Resources/Images/ImagesPMTab.png)

An Image set is a collection of DICOM or other images. Image sets are organized in a tree structure. Top level is the _Patient_. The information shown is retrieved from the DICOM tags (0010,0020) Patient ID and (0010,0010) Patient Name, if possible. When those tags are not available, the string n/a is used. One level below, the _Study_ information is exposed. Those are taken from the DICOM tags (0008,1030) Study description and (0020,0010) Study ID, if possible. When that tags are not available, the string n/a is used.

Image sets are shown in the tab as Image set preview (thumbnail). Each thumbnail contains information about the image set. The name, number of images of the image set and preview of the middle slice are shown.

At the Image set preview (thumbnail) an indicator (blue triangle) is shown to indicate the currently active image set of the project. The active image set corresponds to the image set in which you are working now. There is only one active image set in the project and it is always visualized in the 2D views. If the active image set is the only one visible in the project, then only this image set will have an indicator in the Image set preview (thumbnail), the blue one.

In case two image sets are visible simultaneously, then two indicators are present at the Image set preview. The blue indicator keeps the same meaning and the second indicator (grey triangle) refers to the image set that is only visualized at the 2D views but it is not currently the active one.

**Example** : Only the active image set is visible in the project

![](Mimics Reference Guide/../Resources/Images/ActiveImage.png)

**Example** : There is one active and one visible image set in the project

![](Mimics Reference Guide/../Resources/Images/Active%20and%20Visible%20Image.png)

##### List of the image sets

Name |  Name of the image set. By clicking on the name it can be renamed.  
---|---  
Number of slices |  Number of slices of the image set  
Image set preview |  It shows the middle slice of the image set  
Status indicator |  The blue indicator refers to the active image set. The grey indicator is the visualized image set.   
  
##### Functions on images

Add Images |  Adds image sets in the project  
---|---  
Delete |  Deletes the selected image set  
Image information |  Provides detailed information for the selected image set  
_Matrix View_ | Image sets are shown as a matrix  
List View | Image sets are shown as a list  
Drag and Drop | To visualize an image set in a view, drag it from the Images tab and drop it in a 2D view. I case the view contained an image set before that was the currently active one, then the image set will be replaced by the new one. The new image set will become the currently active image set  
  
##### Image set information

This window displays the image set information along with the DICOM tags.

With the **'Export'** button you can export all the DICOM tags (standard and custom) to a text document.  


![](Mimics Reference Guide/../Resources/Images/ImageInformation_407x504.png)

You can create custom tags tab, clicking the ![](Mimics Reference Guide/../Resources/Images/dicom_tags_add_tag_button.png) button. In the pop window you can set the name of the custom tags tab.

To add a tag in the custom tags tab, right click to the selected tag and select **Add Tag to Tab**.

![](Mimics Reference Guide/../Resources/Images/dicom_tags_add_tag_to_tab_277x79.png)
