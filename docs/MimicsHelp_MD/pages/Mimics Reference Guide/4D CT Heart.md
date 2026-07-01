### 4D CT Heart  
  
The 4D CT Heart is a tool that allows you to automatically segment the anatomical structures of the left side of the heart. The prerequisite to run the segmentation is that one image set is already segmented and the masks that represent the left atrium, the left ventricle and the Aorta are already present..

Click on the 4D CT Heart icon in the Advanced Segment menu to open the following dialog box:

![](Mimics Reference Guide/../Resources/Images/4D_CT_heart_dialog_393x324.png)

_Note_ : The tool will semi-automatically segment image sets that represent the cardiac phases of a 4D CT Cadiac dataset. Before you run the tool make sure that you have opened or imported a 4D CT Cardiac dataset and there is at least one image set segmented.

##### 

The dialog box of the tool consists from three main sections. The _Images selection_ , the **Masks selection** and the **Parameters selection** section.

##### Images selection

Images |  Select the image set that is the source image set. The Images selection section, contains a list of image sets which will be processed and segmented. The list shows only the image sets that belong to the same patient and study as the image sets that is currently active at the time that the tool is launched. In case the Mimics project contains image sets from more than one Patients in the Project Management tab or there are more than one studies under one patient, the images will be ignored. The currently active image set is automatically selected as the source image set when launching the tool. You can use the dropdown menu to select a different image set that will be used as the source image set. This needs to be the image set that is segmented. During the segmentation process, all the images that are present in this list will be segmented in a continuous loop. There is no way to do a step-by-step segmentation.  
---|---  
  
##### 

##### Masks selection

Select the input masks in this section. You can select masks from a dropdown menu. Note that in the dropdown menu are shown only masks that are linked to the image set that you have selected as the source image set. 

Threshold mask |  The mask that defines the threshold range. This mask is used as a reference for the output masks. The output masks will get the same threshold range as this mask. You can select a mask that represents the blood pool or one of the masks below as reference.  
---|---  
Aorta mask | Select the mask that represents the aorta  
LA mask | Select the mask that represents the left atrium  
LV mask | Select the mask that represents the left ventricle  
  
_Note_ : To start the calculation of the segmentation all the fields in this section need to be filled with a mask.

##### Advanced Parameters selection

The Advanced parameters section is by default collapsed. To show the parameters click on the **Advanced** button.

Hole size threshold |  This parameter is used to fill the holes in the masks.   
Range: 1-100  
---|---  
Dilatation Aorta | Aorta mask from previously segmented image set is dilated to be used as restriction area for the next image set to avoid segmentation leakage problems. If these parameters are higher, segmentation results may include surrounding irrelevant parts.   
Range: 1-100  
Dilatation LA | LA mask from previously segmented image set is dilated to be used as restriction area for the next image set to avoid segmentation leakage problems. If these parameters are higher, segmentation results may include surrounding irrelevant parts.  
Range: 1-100  
Dilatation LV | LV mask from previously segmented image set is dilated to be used as restriction area for the next image set to avoid segmentation leakage problems. If these parameters are higher, segmentation results may include surrounding irrelevant parts.   
Range: 1-100  
Seed mask erosion distance |  A large input seed for each mask helps to obtain good segmentation results. If this parameter is higher, then it means more erosion and consequently smaller seeds. It may reduce segmentation quality. And if the parameter is very high, it may erode whole seed and the user will get an error.   
Range: 1-100  
  
##### 

Description of the output

The tool produces masks that correspond to the segmentation of the left side of the heart (LA, LV and Aorta) for all the image sets that are given as input in the tool. For each image set that is successfully segmented, the tool produces exactly three masks that represent the LA, LV and Aorta if this is the given input anatomy. Output masks are linked to the correspondent image set. Resulting masks that are produced by the algorithm are listed in the Project Management Masks tab. The name and color of a resulted mask is the same as the respective mask that is used as input and it represents the same anatomy.

In the image below the masks that are linked to the image set 60%, and more specifically the Aorta, Blood Pool, LA and LV masks, are used as input to the tool. The segmentation algorithm produced set of three masks (LA, LV, Aorta) for each segmented image set that are automatically linked to the image sets that are segmented (70%, 80%..). You can see that the Blood Pool mask with threshold range (229, 3071) defines the threshold range of the output masks.

![](Mimics Reference Guide/../Resources/Images/4d_ct_heart_linked_masks_377x367.png)
