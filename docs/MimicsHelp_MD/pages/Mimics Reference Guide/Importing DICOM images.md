#### Importing DICOM images

When you select a set of DICOM images to import, the New Project Wizard takes you to the following steps. 

##### Organize the images

In this window information is shown about the studies you are going to open in Mimics. In the preview pane on the bottom you can browse through the images of the selected serie and check all relevant header information. You can resize the different blocks using the mouse to adjust the window appearance.

![](Mimics Reference Guide/../Resources/Images/New_project_wizards_528x450.png)

The sets of images are organized in a table with a hierarchical structure following the following logic:

  * A Patient is created on the top level. If there are more than one patients then more top levels are created
  * A Study is created in a level below. Under each patient you can find one or more studies depending on the scans and on the grouping preferences.
  * The lower level is the series level. Under each study you can find one or more series.


The columns of the table can be customized. To add or remove columns from the table click on the button **Customize Columns**.

![](Mimics Reference Guide/../Resources/Images/customize_columns_150x212.png)

In the dropdown list the latest used DICOM tags are shown. You can include or exclude a DICOM tag in a column using the check box in the left side of each DICOM tag name. To remove an entry from this list, click the ![](Mimics Reference Guide/../Resources/Images/x_button_customize_columns.png) button. To add new DICOM tags in this list, click on the Button **Add DICOM tags** on the bottom of the list.

The following window will appear where you can select the DICOM tags to add in the original dropdown list. To select a DICOM tag click on the checkbox . To add it in the dropdown list, click Add.

![](Mimics Reference Guide/../Resources/Images/customize_columns_window.png)

To add new sets of images in the existing table click on the button **Add**. A new window will appear to guide you to select your folder with images. The new sets of images will be appended in the existing table under the correct patient and study.

If multiple image sequences of the same patient are displayed, this means that not all image parameters are equal. To merge one or more sequences to one Mimics project, select them by using the SHIFT key and left mouse button, and click the Merge button . Merging sequences is not allowed when the patient name, pixel size or image orientation is different

To remove an entry from the table (it can be patient(s), study(s), serie(s)) select it and click the **Delete** button

All sets of images that are selected will be imported and opened in Mimics.

To unselect an image set, click on the checkbox in front of each study or on the checkbox on top of the images in the Series Preview Pane. Click the Open button to open the files images in Mimics..

Mimics also performs a check of the memory needed by the project files along with the available memory. If the available memory is less than that required, Mimics will give a warning on the bottom left of this window and you are requested to free up memory in your computer, use a different computer, or compress your data.

A description of the different tags and options on this step of the wizard follows: 

###### a. DICOM tags

In the lower right, you can view the information relevant to a particular series. These are grouped according to following different tags:

All | This tab contains all tags available in the file.  
---|---  
Acquisition | This section lists the parameters or scanner settings used to acquire the images.   
Critical | This tab contains parameters that are critical for Mimics to load the images.  
Image | This tab lists out parameters of the imported images.  
Main | These are the main parameters of the DICOM header.   
Patient | This tab contains the patient specific parameters.   
Voluson | This tab contains the set of tags of VOL format files   
  
![](Mimics Reference Guide/../Resources/Images/Dicom_tags.png)

You can create custom tags tab, clicking the ![](Mimics Reference Guide/../Resources/Images/dicom_tags_add_tag_button.png) button. In the pop window you can set the name of the custom tags tab.

To add a tag in the custom tags tab, right click to the selected tag and select **Add Tag to Tab**.

![](Mimics Reference Guide/../Resources/Images/dicom_tags_add_tag_to_tab_277x79.png)

**Note** : The slice increment as identified in Mimics is not taken from the DICOM tag (0018, 0088: Spacing Between Slices), as this tag is often not filled or filled incorrectly, but calculated from the difference in image positions between 2 neighboring slices.

###### b. Grouping

Mimics checks for several parameters during the import of images. If one of these parameters is different, Mimics splits the data set in different parts. You can select or unselect the parameters that you wish Mimics to consider while performing the check, as shown below. The parameters that Mimics checks for are: patient�s name, series description and study description. 

![](Mimics Reference Guide/../Resources/Images/image27_243x306.png)

There is also a check box for Image center which is unchecked by default. However, when DICOM images are imported which are not aligned, these default settings will generate the following warning message:

![](Mimics Reference Guide/../Resources/Images/image57.png)

If you choose �no�, you will be able to restart:

![](Mimics Reference Guide/../Resources/Images/image58.png)

As a result, you will be able to check the Image center as one of the parameters to divide the data in different parts.

In general, the information from the Patient�s name, Series Description and Study description is available in the DICOM tags. The Image center parameter is derived from the image orientation and image position DICOM tags. 

c. **Log**

![](Mimics Reference Guide/../Resources/Images/log_import_wizard_491x321.png)

In the log panel you can find an overview of the images imported in different steps. In case a file could not be imported, it's full path will be shown in the log panel. 

To copy the log click on the **Copy log to clipboard** button. To save log click on **Save log**.

d. **Open images in Mimics**

Press 'Open'.

If the pixels in the dataset are rectangular you will receive the following message:

![](Mimics Reference Guide/../Resources/Images/Rectangular_Pixels2.png)

If you chose �Reslice images�, the anatomical proportions will be preserved, however the dimensions and greyvalues of the dataset will be recalculated and interpolated. It is recommended to use when the difference between the sides is rather big (e.g. 0.5x0.7).

If you chose �Resize images�, initial grey values will be preserved, however the dataset will be visually stretched and measurements will be influences. It can be picked if the deviation between width and height is very small (e.g. 3.9999999 and 4.0).

You can select the desired pixel size using the dropdown box. The action will be applied after you press 'OK'. If the checkbox �Apply to all series with same pixel sizes� is checked, the settings will be repeated for all series selected in the Import Wizard.
