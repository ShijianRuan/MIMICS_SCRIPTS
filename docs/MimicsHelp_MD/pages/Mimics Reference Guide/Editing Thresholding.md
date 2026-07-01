#### Editing - Thresholding  
  
##### Separating maxilla and mandible

To separate the mandible from the maxilla, we have to disconnect them manually. Therefore we erase a layer from the active mask somewhere between the mandible and the maxilla. Then we perform a region growing on the mandible. The result is that both mandible and maxilla will be in a different mask and thus separated.

Look at the sagittal image and place the horizontal indicator between the maxilla and the mandible. Note that it will not be possible to separate them correctly in every image, so we have to find the best possible position.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300029D_448x298.png)

In the corresponding axial image all pixels have to be removed from the active mask. The position of the axial image corresponding to the position of the horizontal indicator in the figure above, is -4.50. Go to this image and press the Edit masks button ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0200029E_25x25.jpg). Select the Erase mode, choose a big square as type of cursor and remove all pixels from the active mask. Make sure you don�t forget any! Go to a lower image in the data set and do a region growing of the mandible (do not activate the Leave Original Mask option). Now you have two masks, one for the mandible and another for the maxilla.

Note: In the region growing toolbar, if you activate the Leave Original Mask option, the pixels selected with region growing will be put into a new mask, but they will also remain in the original mask. If the result of the region growing is not satisfying, you still have the complete original mask and you can start over. If this option is not activated, the pixels selected during region growing are removed from the original mask. In this case you can�t do the region growing again from the same original mask.

Change in the Project Management the name of the two masks to �mandible� and �maxilla�, respectively. In figure below, these two masks are shown and the red line in between indicates the layer that was removed from the active mask.

But be careful! As it was not possible to perform complete separation between mandible and maxilla, therefore, we will still have to edit the images and make sure that all the pixels that belong to the mandible are really in the mandible mask.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300029F_374x214.png)

Scroll through the coronal images and check if every pixel that belongs to the mandible is in the proper mask. Do you notice at position 64.50 that some pixels (at the left side in the image) from the maxilla are wrongly put in the mask of the mandible? 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002A0_448x277.png)

Move both indicators until their point of intersection indicates the wrong pixels (figure above). It concerns two layers of pixels, belonging to a tooth of the maxilla. In the two corresponding axial images (position -6,50 and -5,50), erase the tooth from the mask of the mandible. You cannot be mistaken, because that tooth is also indicated with the point of intersection of the indicators (figure below). If the two layers of pixels are shown in gray values in the coronal image, you can be sure you erased the whole tooth from the mandible mask. If not, move the indicators again in the coronal image so their intersection points to the wrongly colored pixels. In the axial image, remove the pixels that are indicated by the indicators from the active mask.

Note: you can still access the 1-click navigation function by pressing the SHIFT button while you are editing. You can then click with your left mouse button on the point you want to navigate to.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002A1_187x187.png) ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002A2_187x187.png)

Axial images (position: left -5,50, right -6,50): the indicators point out the tooth that does not belong to the mandible mask.

In the sagittal image at position 87.25 (or the coronal image at position 43.25) another collection of badly masked pixels is visible. But now it�s the opposite situation! Three layers of pixels that belong to the mandible are not in the mandible mask. Two layers belong to the maxilla mask and the other layer is the one we erased in the beginning to make the disconnection. Again, mark these pixels with the indicators as it is done in the figure below. In the corresponding axial images (at positions -4.50 and -3.50 and -2.50) the pixels (of a tooth) should be added to the mandible mask. We will make use of a local threshold to do this. To make this threshold clear, a short intermezzo is inserted below.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002A3_334x213.png)

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002A4_448x270.png)

Indicators point to pixels that should belong to the mandible mask (sagittal view).

Local threshold: In the obturator case it is mentioned that there are three modes to choose from in the Edit toolbar i.e. draw, erase, and threshold. The threshold mode (Ctrl + T) is used to set a local threshold. This means that if you apply a local threshold in a particular area of one image, this threshold doesn�t apply to the other images in the project. Remark that the threshold we�ve set in the beginning of this case was global and it applied to every image in the dataset.

When you activate this mode, the box with the two default threshold values is shown on your screen. To set a different local threshold, press one of the two arrow buttons and double click on a threshold value. After you changed the value, press Enter. When you move the square over the image while pressing the left mouse button, every pixel that comes to lie within the square and has a threshold in the threshold range you just set, will be added to the active mask. On the other hand, all the pixels that already belonged to the active mask and that don�t have a gray value within the range will be removed from the mask.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002A5.png)

The local threshold range

For the moment we don�t have to change the threshold values, but it will be used later on in this case to remove artifacts out of the image.

Maybe you now wonder why we will add the pixels of the teeth that belong to the mandible with this local threshold method and not with the draw mode we will use in the obturator case. With the draw mode, can�t you also add pixels to a mask? Yes, that�s true, but there is a difference! With the draw mode you add every pixel you touch with your cursor. With the threshold mode you do the same, but there is one more condition before they are really added: their HU values must lie in the range shown in the box (figure 3-7). In this case, it�s much safer to add pixels by taking into account their gray values. Our segmentation will be more accurate.

Press Ctrl + T. The Edit toolbar shows up and the threshold mode is already selected. Choose a circle as type of cursor and make it more or less the same size as a tooth. Make sure that the mandible mask is the active mask. Press the left mouse button and go over the tooth with your cursor. Make sure you got the tooth completely. You can check this very easily by looking at the sagittal or coronal image: if the wrongly masked layers (figure 3-6) now have the color of the mandible mask it�s alright, otherwise you�ve forgotten some pixels. Suppose you added too much pixels, just press E (or select the Erase mode with your mouse) and erase them. If you repeat the thresholding in the necessary axial images (see before to know their positions) you should have a sagittal image like in the figure below. Now we can say that the whole mandible is in the mandible mask.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002A6_448x271.png)

Sagittal image after local thresholding

##### Artifacts

The images still don�t look nice, because of all the artifacts. We are going to get rid of them by again performing a local threshold, but not the default one like we just used to add the pixels.

To enter the Edit mode, press Ctrl + T. The threshold mode is already selected. Click on the top arrow of the threshold range box (figure 3-7) and double click the threshold 1 value. Change this value to 3000 (if you are working in Hounsfield Units) and press Enter. Because the Hounsfield Units of the artifacts are lower than the ones of the teeth. Go with your cursor over the artifacts and notice that they disappear. Why do we use this high local threshold? Because the HU values of the artifacts are lower than the ones of the teeth. So by setting a very high threshold the artifacts will be removed from the mask because their gray values are not in the range. Moreover, if you accidentally go with your cursor over the teeth, their pixels will remain in the mask, except for the edges (their HU are lower). If you removed the edges from the mask, don�t panic. Set the threshold range back to the default one by clicking once on the lowest arrow and move your cursor over the tooth again to restore the edges. So, this is the way you should work. Scroll through the axial images and remove all the artifacts from the mask of the mandible. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002A7_214x177.png) ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002A8_215x177.png)

The artifacts in the left image are removed with a local threshold. The right image shows the result

##### Multiple particles

Let�s calculate the 3D image of the mandible. Press the Calculate 3D button and select the mandible mask to be calculated (choose low quality). You get the message that the mask consists out of multiple parts. Answer �Yes�.

![](Mimics Reference Guide/../Resources/Images/Editing-Thresholding.png)

Visualize the 3D by pressing the 3D button. Rotate the model and remark that there are little particles floating around the mandible due to which you got the message about the multiple parts. The particles are due to the editing you�ve done to remove the artifacts. To avoid this you have to do a region growing before calculating the 3D. Press the 3D view button again to get back the sagittal image. Press the Region grow button and click into the mandible. Change the name of this new mask to �Total mandible�. Now calculate and visualize the 3D model of the final mandible. You can delete the first 3D (with the particles) listed in the 3D tab of the Project Management.

##### Scan prosthesis

Can you distinguish between the natural teeth and the scan prostheses in the 3D model of the mandible? It�s quite simple; the natural teeth are connected to the bone, while the scan prostheses are not. There are 3 teeth of the scan prostheses at the patient�s left side and one at his right side. In the figure below, the scan prosthesis (axial view) is marked with rectangles.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002AA_315x226.png)

Axial image indicating the prosthesis in the boxes.

We would like to have the mandible without the prosthesis and the prosthesis itself into two different masks. There are two ways to achieve this. The first one is to proceed with the segmentation of the final mandible and to remove the prosthesis from the active mask. The second option is to perform a segmentation of the prosthesis. We opt for the latter. We will do a region growing of the prosthesis twice, once at either side. But, we first have to make sure that the prosthesis is completely disconnected from the natural teeth. The intention is to remove (from the final mandible mask) the pixels surrounding the prosthesis and the pixels connected to the prosthesis. The goal is to get the prosthesis nicely isolated in every image. Keep the following advice into account: remove enough pixels in the surrounding of the prosthesis, because sometimes in 2D it looks like there is no connection, but there is still one in 3D. So a 3D model can be very tricky!

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002AB.png)

Project Management � Masks tab

Make the mask of the final mandible active and press the Duplicate![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/020002AC_20x22.jpg) button in the Masks tab of the Project Management window. This way a backup mask is created that we can use to do the segmentation of the prosthesis, while the original Final mandible mask is left unchanged. The original one will be used later on to perform Boolean operations. Proceed with this backup mask (if you don�t like the color, press the Color button in the masks tab and choose the color you like). Scroll through the axial images and remove (enough!) pixels surrounding the prosthesis from the active mask. In the figure below it is shown for the axial image at position -11,50.

If you think you disconnected the prosthesis completely, press the Region Growing button. Make sure your target mask is a new mask (if not, select �new mask� from the drop down list) and that you activate the Leave Original Mask option. This last option is very important!

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002AD_313x217.png)

The prosthesis is disconnected in this layer.

Click on the left or the right prosthesis. If you disconnected the prosthesis entirely, only the prosthesis should be shown in the color of the target mask. If this is not the case, make the previous mask active again, delete the last mask in the list (generated for the region growing) and remove more surrounding pixels from the backup mask. Also in the layers where you don�t see the prosthesis it can be useful to remove some pixels belonging to the teeth next to the prosthesis. repeat these actions for the prosthesis at the other side. Give the masks of both prostheses proper names.
