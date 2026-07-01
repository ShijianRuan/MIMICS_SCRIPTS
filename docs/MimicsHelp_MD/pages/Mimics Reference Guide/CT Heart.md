# CT Heart

The CT Heart tool is part of the _Cardiovascular_ menu and it allows semiautomatic segmentation of heart chambers based on CT data. The Cardiovascular menu is accessible via the Segment menu and is only visible when the C&V Segmentation module is activated. In the toolbar, it can be found under the Cardiovascular tab.

When CT Heart is initiated, the following dialog box appears on the screen.

![](Mimics Reference Guide/../Resources/Images/CT_Heart_Dialog.png)

The first step is to select the region of interest by adjusting the crop box on the image views.

![](Mimics Reference Guide/../Resources/Images/CT_Heart_Cropbox_621x347.png)

The next step is to select the desired threshold by changing the sliders or entering the minimum and maximum values manually. With the shortcut, the sliders of the histogram can be moved to interactively change the threshold values and immediately review the colored pixels on the images. Press the shortcut SHIFT + CTRL on the keyboard and the left mouse button for the Minimum threshold value. While pressing down, move the mouse to change the histogram slider. For the Maximum threshold values do the same but use the right mouse button. The initial mask is updated accordingly when the threshold values are changed.

Once the threshold is selected, press '**Calculate** '. A percentage of completion will be displayed in the progress bar as the masks are calculated.

![](Mimics Reference Guide/../Resources/Images/CT_Heart_ProgressBar.png)

After segmentation is completed, the resulting temporary masks are displayed in the Current Active Masks list.

![](Mimics Reference Guide/../Resources/Images/CTHeart_CurrentActiveMasks.png)

If the result is acceptable, press 'Close'. After this action the resulting masks are displayed in the Project Management tab.

![](Mimics Reference Guide/../Resources/Images/CTHeart_PMTab.png)

Editing masks

After the masks have been created, it is possible to verify the masks by scrolling through the images and using 3D preview by selecting the temporary mask in the list and ensuring that the 3D preview button is turned ON. 

![](Mimics Reference Guide/../Resources/Images/CTHeart_3DPreview.png)

In case the masks are not optimally segmented, it is possible to correct the masks in the following ways.

_Changing threshold_

The result of the semi-automatic segmentation can be suboptimal if an inappropriate threshold is selected. A new threshold can be selected which will generate new masks. 

**Adding seed points**

When the masks are generated and you defined that the specific mask corresponds to the specific chamber, you can use seed points to assign specific chambers to a specific masks. In specific scenarios it is possible that part of the mask is either missing or is part of another mask. Seed points can then be added on the images to generate the corrected masks, corresponding to the desired anatomical part.

![](Mimics Reference Guide/../Resources/Images/CT_Heart_SeedPointsTable.png)

The left column contains the list of generated masks while the right column contains a tabbed structure for the addition of the seed points.

The seven tabs allow to indicate seed points for the heart chambers, aorta, and the pulmonary artery:

  * LA � Left Atrium
  * LV � Left Ventricle
  * RA � Right Atrium
  * RV � Right Ventricle
  * Aorta
  * PA - Pulmonary Artery
  * Other


Other group can be used differently. If you need to separate anatomical parts not related to chambers in a separate mask, or there�s a chamber of the heart that cannot be certainly defined as a specific chamber (e.g. in congenital heart cases), you can place seed points there to define this area.

Pressing the **Add** button will add a seed point for the selected tab. It is possible to change the radius of the seed point by modifying the Radius parameter or changing it interactively on the 2D views.

It is possible to remove a single seed point with the **Remove** button or remove all seed points at once with the **Remove All** button.

The seed points can be located on the image views by using the **Locate** button.

As soon as a seed point is marked on the images, it appears on the list of seed points of the corresponding chamber. After the seed points have been defined on the images, press **Calculate** to generate the new masks.

Only the masks on which the seed points have been placed are considered during recalculation.

Alternate workflow

Alternatively, you can also start the CT Heart tool, use the crop box to define the region of interest and immediately place seed points for chambers under the respective tabs, and calculate the masks. The resulting masks will have the respective colors and names of the chambers.

Hint: if you need to segment only one chamber, you can place one group of seeds to this specific chamber and another group of seeds to Other. In such a way the result will be separated in two masks: your desired chamber in one and all the rest in another one.

After the masks are reviewed and accepted, press **Close** to exit the CT Heart tool. The masks generated from the tool are saved in the Project Management tab; all existing operations are possible on the resulting masks.
