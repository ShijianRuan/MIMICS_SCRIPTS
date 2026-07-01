# Mimics Tutorial

*Tutorials*

Below, you can find a selection of tutorials to guide you through example workflows utilizing some of the tools that Mimics provides.

| Chapter                                 | Description                                                  |
| --------------------------------------- | ------------------------------------------------------------ |
| Chapter 1: Import                       | Tutorial that shows how you can import images in Mimics      |
| Chapter 2: Mimi                         | Tutorial that shows how to do a basic segmentation and 3D calculation. |
| Chapter 3: Simon                        | Tutorial that shows some advanced segmentation functions to remove artifacts. |
| Chapter 4: Smart expand                 | Tutorial that shows how to apply the smart expand to segment a liver. |
| Chapter 5: Hip                          | Tutorial that shows how to use the Analyze module.           |
| Chapter 6: Obturator                    | Tutorial that shows how to make a mold of a cavity by segmenting the Soft tissue around the cavity. |
| Chapter 7: Manual Import                | Tutorial that shows how to use the manual import function.   |
| Chapter 8: FEA Tutorial                 | Tutorial that shows how to use the FEA module.               |
| Chapter 9: Simulation Tutorial          | Tutorial that shows how to use the Simulation module.        |
| Chapter 10: CFD Tutorial                | Tutorial that shows how to use the FEA module for linking to CFD. |
| Chapter 11: Non-manifold <br>Assemblies | Tutorial that shows how to combine two meshes.               |

> **Note:** In Mimics you have the possibility to use both Hounsfield and Grey Values. This is very important when setting a threshold and when you use the Profile Line function. To switch between these two possibilities, go to Edit > Preferences, General tab and select the Pixel unit you want to use. Most tutorials need one or more modules of Mimics (STL+, RP Slice, Analysis, Simulation or FEA). If you wish to try that section of the tutorial and you don't have the required module(s) installed, an evaluation period of that module can be obtained on request.

**Datasets**

If you have chosen to install the demo files during the Mimics installation procedure, the files used in this tutorial will be put in the MedData folder.

---

## Import

The goal in the first part of this chapter is to teach you how to import images and convert them into a Mimics project. The second part will illustrate how to organize the images in the project you made.

In this tutorial we will discuss three topics:

- How to do an Automatic Import
- How to organize images
- How to do a Semi-Automatic Import

> **Note:** There are 3 ways to import images, depending on their format:
> 1. automatic import, when the format of the files is known to Mimics
>    - import in strict mode, which strictly complies to the DICOM 3.0 standard
>    - non-strict mode, which doesn't enforce DICOM tags to be conformant with the DICOM 3.0 standard
> 2. semi-automatic, e.g. Bitmap or Tiff images
> 3. manual import (**Import Raw Images**), when the file type is unknown and you need to specify some parameters manually

### Automatic import

To start the Import wizard, first select **File** and then choose **New Project Wizard**. In the File Browser window, you can select where the images to be imported can be found (STEP 1).

Browse to the MedData folder and select the folder called "DICOM_Mandible" in the File browser. The list of files will be displayed in the Filename column and all the files will be automatically selected. Click on one of the files and press CTRL+A to select all files in that folder. Click the **Next** button.

![New Project Wizard](Resources/Images/New_Project_wizard_707x416.png)

An Import log window will show details of the import, including the recognized formats of the file (see the general help files for a list of known formats).

![Import log](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000278_554x343.png)

> **Note:** In this case the file type is recognized, but in some cases the message log tells you that one or more images are of an unknown file type. If this happens, you have to perform a manual import (see **Import Raw Images**).

Click **Next** to proceed to the Studies page. This is the second step of the New Project Wizard, in which you need to select the studies to be converted. The study you have just imported is already selected by default.

In this window some information about the project can be found, such as the number of images, pixel size, patient name, orientation parameters, etc. You can also compress your studies to cut off unwanted regions like Air. For this case, we will chose Lossless Compression.

![New Project Wizard 2](Resources/Images/New_Project_wizard2_431x366.png)

Click **Open** and you will see a progress bar. After the images are successfully imported, you will see a Check Orientation window where you can check and change the orientations of the imported study. Here, the orientation strings L and R stand for Left and Right, A and P stand for Anterior and Posterior, and T and B stand for Top and Bottom respectively. To change the orientation, click on one of the letters and chose the correct orientation from the list. Note that all the other orientation strings are updated automatically.

![Check Orientation](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/0300027A_346x264.png)

If some orientation is not defined in the DICOMs, you will see an X mark, indicating a missing orientation. You can click on the X mark and assign an orientation to it.

If the orientation of the images is correct, click **OK** and your Mimics project will open. Now you can process your images using the tools explained in the tutorials **Simon** and **Smart Expand** (Threshold, Region Growing, Edit, etc.)

### Organizing images

Once you have opened your project, you can decide to exclude some images if they are not good or if you don't need all of them. For example, we can decide to delete the images of the project Simon.mcs that don't include parts of mandible or that don't contain any information.

To access the Organize Images window, go to **Image** and then choose **Organize Images**.

![Organize Images](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/0300027B_498x399.png)

You can get a better look at the images by changing the preview size to Medium or Large by selecting the respective size from the **Preview size** dropdown box.

If you look at the images you will notice that the ones that correspond to table positions -9.5 and -8.5 do not contain any information about mandible. So you can click on these two images to unselect them, the green mark will disappear and the image will be unchecked in the list on the left.

You may also notice that the image at table position -0.5 is the last one that contains information about the mandible. Right-click on the image at position -0.5 and choose Unselect after this. All the consecutive images will be unselected also.

Press **OK** and scroll through the axial images to check if the correct ones are visible in the project, you should not see the images which were unselected.

You can now save your Mimics project with the name "Organizing Images.mcs" by going to **File** and then **Save As**. After you have done this you can make a segmentation following the next tutorial (**Simon**).

### Semi-automatic import

Now we will try to import the Bitmap images, you can find the dataset in the folder "BMP_Leg" in your MedData directory. Select **File > New Project Wizard** and browse to the C:\MedData\DemoFiles\BMP_Leg directory. Click on one of the images in BMP_Leg folder and press Ctrl+A on your keyboard to select all files in it. Press the **Next** button and the Import Log will be displayed. Click **Next** to see the Images Properties dialog.

![Images Properties](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/0300027C_554x343.png)

Here you can preview your images and order them according to your preferences. You can also check if the scan resolution is correctly read. Uncheck force isotropic sampling checkbox and change the Z direction to 1. You can also change the dimensions of your images. This information will be typically provided by the radiologist who took the scan. Correct values should be entered here to ensure correct dimensions of the volumes and the Parts that will be created further on. Leave it in mm scale for this case and click **Next**.

![Edit Images](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/0300027D_554x343.png)

In the Edit Images dialog, you have the option to crop the images or resample them. For this example, we will leave the values as is and click **Next**.

Now you should be able to set the orientation parameters as described in the previous paragraph and calculate a good 3D.

---

## Mimi

In this tutorial we will show you some basic features of Mimics. The topics that will be discussed are:

- Opening the Project
- Windowing
- Thresholding
- Region Growing
- Creating a 3D representation
- Displaying a 3D representation
- View of the end result

### Opening the project

From the **File** menu, select **Open** (Ctrl+O). The Open dialog box shows all projects in the working directory. Double click on the Mimi.mcs file (Mimics project file).

All images are loaded and displayed in three views. The view on the right shows the images as they are exported by the scanner (xy-view or axial view). The upper left corner is a reslice of these images in the xz-direction (xz-view or coronal view) and the bottom left is a reslice in the yz-direction (yz-view or sagittal view). The different colors of the intersecting lines refer to the colors of the contour lines of each view so every line refers to the slice in the corresponding view. You can easily navigate through the images by clicking on any point of the CT images in any view: the intersecting lines will move crossing each other in the point you clicked and all the views will be updated showing the corresponding slices.

![Mimi view](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/0300027E_554x298.png)

If you need to change the orientation of a view, go to **Image > Change Orientation**. This will open a window in which you can change the orientation parameters simply by clicking on it with the right mouse button (see **Import**).

![Change Orientation](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/0300027F_488x372.png)

In the Mimics window, you will see several indicators, intersection lines, tick marks etc. To deactivate an indicator, go to View > Indicators in the Menu Toolbar, and toggle them off.

In the right border of the window you will see a slider that allows you to scroll through the images from the active view.

In our current project (Mimi), all images are correct. If, however, you have an image set from which you want to remove some images, go to **Image > Organize Images**. There you can add or remove images (see **Import**).

### Windowing

First of all, we have to adjust the contrast of the images displayed in the different views. Contrast enhancement is a very good tool for selecting parts with different intensities, e.g. bone vs. brain tumor. This action can be performed at any time.

You can change the contrast in the corresponding tab of the Project management. The contrast tab shows the histogram of the project with a line representing the "window". The gray values or Hounsfield units below the start point of the line will be displayed in black. All gray values above the end point of the line will be displayed in white. The gray values in between the window will be mapped on a shade of gray. You can change the window size by clicking your left mouse on one of the points and dragging it to its new location. To move the window select the line and drag it to its new position. You can also choose one of the predefined "windows" by selecting the appropriate scale from the menu on the bottom of the tab.

The following steps will describe the necessary actions to achieve a nice segmentation mask. A segmentation mask is a collection of pixels of interest that constitute an object you wish to work on. One can create several - dependent or independent - masks, each displayed with their own identifying color. Usually several masks will be needed to obtain a final segmentation object that contains the information that is needed.

### Thresholding

Thresholding means that the segmentation object (visualized by a colored mask) will contain only those pixels of the image with a value higher than or equal to the threshold value. Sometimes an upper and lower threshold is needed; the segmentation mask contains all pixels between these two values.

> **For example:**
> A low threshold value makes it possible to select the Soft tissue of the scanned patient. With a high threshold, only the very dense parts remain selected. Using both an upper and a lower threshold is needed when the nerve channel needs to be selected. Defining a good threshold value also depends on the purpose of the model. If you just want a nice looking model, a lower threshold value is recommended since it will result in a model with fewer holes. On the other hand, when the model serves for modeling prostheses a higher threshold value is preferred.

- Click the **Threshold** button ![Threshold icon](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/020000BC_25x25.jpg):

To change the threshold value, press the left mouse button on a slider in the Threshold Toolbar and move the slider by moving the mouse (while still holding the left mouse button).

Some tips for selecting an adequate threshold value:

Look at different images. You can change images of any view by:
- using the arrow keys, the page up and page down keys
- using the slider on the right in the window border
- moving the slice indicators

- Click the **Draw Profile Line** button ![Profile Line icon](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/02000280_25x25.jpg):

In the axial view draw a line over the bone as shown below. To draw this line, click the left mouse button in the soft tissue to indicate the starting point, move the mouse over the bone click. Along this line an intensity profile is generated. The straight horizontal lines represent your current threshold value. Click on **Start Thresholding** and drag the lower straight-line up/down to set a good threshold. If you want a good visualization model, select a threshold slightly above the intensity plateau of the soft tissue. If your model will serve for modeling prostheses, place the line between the soft tissue plateau and the top value of the bone. If a proper threshold is set, click on **End Thresholding** to save the current value.

| Profile line drawing                                         | Profile dialog                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![Profile line drawing](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000281_183x204.png) | ![Profile dialog](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000282_262x181.png) |

Zoom in on a part you're interested in. First, from the pull-down menu next to the zoom button, select Box. Click the **Zoom** button ![Zoom icon](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/02000283_25x25.jpg): the mouse is displayed as a loupe. Click the left mouse button on the image and drag for creating a zoom rectangle, release for zooming. To return to the whole image, click the **Fit to screen** button ![Fit to screen icon](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/02000284_25x25.jpg).

A good threshold value for Mimi is about 270 (Hounsfield scale). The threshold value is displayed in the Min. box of the Threshold toolbar. To end thresholding, click the **Apply** button.

After the thresholding operation a green mask will be created. In a project you can have different masks but you can use the segmentation tools only on the active mask. To choose the active mask, select it in the mask tab in the project management. In case the project management isn't active, select the project management button ![Project Management icon](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/02000285_25x25.jpg) in the main toolbar.

![Mask tab](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000286_236x217.png)

You can also hide any mask by clicking on the eye of the corresponding color.

### Region growing

The region growing tool makes it possible to split the segmentation created by thresholding into several objects and to remove floating pixels.

Click the **Region grow** button ![Region grow icon](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/020000BD_25x25.jpg) or press Ctrl + R. The mouse is now cross-shaped and the Region Growing window is on the screen.

Select the Source (= Green) and Target mask (= New Mask). Click the left mouse button on one point in the green area of the object of interest (which is a part of the current segmentation object, i.e. part of the skull). The program starts to calculate the new segmentation, all points in the current segmentation object that are connected to the marked point will be used to form a new mask. The new segmentation is colored yellow.

Click the **Close** button to close the Region growing window.

To make this new mask active, select "Yellow" in the Visualization toolbar. Clicking on the green glasses will hide the green mask. Clicking the button again will make the green mask visible.

Check the mask on different images. When we check the images, we see that everything looks fine. It's time to build a 3D representation.

> **Note:** Thresholding needs to be done before region growing, since all previous work is lost after changing the threshold value.

### Creating a 3D representation

In the mask tab you see all created masks listed with their respective threshold. The names of these masks are Green and Yellow. Selecting one mask will make it active.

Now, you still know that the Yellow mask contains the skull, but after a month, when you reload a project, it might be difficult to know in which mask your end result was stored. Therefore, it is advisable to rename the mask (in Project Management, Masks tab). Click on the name Yellow so that it becomes editable; replace Yellow with a more telling name like "skull".

![Rename mask](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000287.png)

Click on the **Calculate Part** button ![Calculate Part icon](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/02000288_25x25.jpg).

The **Calculate 3D Models** dialog box is displayed. Here you can select from which masks you want to calculate the 3D model. To select multiple masks hold the Ctrl key while selecting the other masks. In this case select "skull" and press the **Calculate** button to generate a Part.

You can set the visualization quality of your model. This is only the visualization on the screen; this parameter does not have any impact on the model that you will actually build on a RP machine!!! Of course, the lower the quality, the less time the program needs to calculate the 3D image and the less memory is needed to load the 3D image afterwards.

### Displaying a 3D representation

In the vertical 3D toolbar on the right, you can set the visibility of the different calculated 3Ds. This can also be done in the Project Management's 3D Objects Tab, by clicking on the glasses.

Once the 3D image is loaded, different operations are available:

- rotate the model with the button ![Rotate icon](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/02000289_25x25.jpg) on the right of the 3D window or moving the mouse pressing the right button;
- select different standard views, like Top, Front, Bottom, by clicking on the button ![Views icon](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/0200028A_20x17.jpg) on the right of the window;
- zoom with the Zoom buttons or Pan with the ![Pan icon](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/0200028C_25x25.jpg);
- change the color of your model by clicking the right mouse button and selecting the option "Color";

The transparency of the model can be changed. To do so, push the **Toggle Transparency** button ![Toggle Transparency icon](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/0200028D_25x25.jpg)

To change the background color, go to **File > Preferences > Visualization** and select the color you prefer.

### View of end result

![Mimi result](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000298_300x236.png)

---

## Simon

The Simon case is an example of a dental segmentation. The mandible of the patient was partially edentulous and needed a prosthesis. First, a scan prosthesis was made that resembled the new teeth to be implanted. The patient had this prosthesis at the correct position in his mouth during the CT scan. Because scan prostheses are made out of barium sulfate, an opaque material, they are clearly visible in a CT image. The result is that you see both the bone and the prosthesis in one image, well positioned against each other. Such a procedure with a scan prosthesis gives better esthetic results and the surgeon is able to make a better planning.

The images in the Simon project are CT scans of the jaw together with the scan prostheses. It will be your job to do the segmentation of the mandible and the prosthesis.

The topics that will be discussed are:

- Opening the Project
- Preparation of the data
- Windowing
- Thresholding
- Region growing
- Editing
- Artifacts
- Multiple particles
- Scan prosthesis
- Boolean Operations
- View of the end result

### Opening the project

In the **File** menu, select **Open** (Ctrl+O). Double click the Simon.mcs file.

#### Windowing

For correct windowing see the windowing procedures in "**Mimi**".

#### Thresholding

Go to an axial image where the mandible (without the teeth) is visible (for example, at position -30.50). Press the **Profile line** button and draw a line over the bone. The figure below shows a profile line and the corresponding profile dialog box. Press **Start thresholding** and drag the threshold line to a value of about 538 (Hounsfield scale). End the thresholding and save your settings. Close the dialog box.

![Profile line bone](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000299_274x214.png)

![Profile line dialog Simon](Resources/Images/Profile_Line_588x483.png)

*Profile line over the bone (upper image) and the corresponding profile dialog box*

#### Region growing

Press the Region Growing button ![Region grow](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/0300029B _22x22.png) and click on the bone of the skull to start the region growing. The skull is now added to a new mask. Click on the Project Management icon ![PM icon](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/0200029C_25x25.jpg). In the Masks tab, double click the name of the mask and change it to "skull". Make the previous mask invisible (make sure the skull mask is active before making the first mask invisible).

#### Editing - Thresholding

##### Separating maxilla and mandible

To separate the mandible from the maxilla, we have to disconnect them manually. Therefore we erase a layer from the active mask somewhere between the mandible and the maxilla. Then we perform a region growing on the mandible. The result is that both mandible and maxilla will be in a different mask and thus separated.

Look at the sagittal image and place the horizontal indicator between the maxilla and the mandible. Note that it will not be possible to separate them correctly in every image, so we have to find the best possible position.

![Sagittal separation](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/0300029D_448x298.png)

In the corresponding axial image all pixels have to be removed from the active mask. The position of the axial image corresponding to the position of the horizontal indicator in the figure above, is -4.50. Go to this image and press the **Edit masks** button ![Edit masks](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/0200029E_25x25.jpg). Select the **Erase** mode, choose a big square as type of cursor and remove all pixels from the active mask. Make sure you don't forget any! Go to a lower image in the data set and do a region growing of the mandible (do not activate the **Leave Original Mask** option). Now you have two masks, one for the mandible and another for the maxilla.

> **Note:** In the region growing toolbar, if you activate the **Leave Original Mask** option, the pixels selected with region growing will be put into a new mask, but they will also remain in the original mask. If the result of the region growing is not satisfying, you still have the complete original mask and you can start over. If this option is not activated, the pixels selected during region growing are removed from the original mask. In this case you can't do the region growing again from the same original mask.

Change in the Project Management the name of the two masks to "mandible" and "maxilla" respectively. In figure below, these two masks are shown and the red line in between indicates the layer that was removed from the active mask.

But be careful! As it was not possible to perform complete separation between mandible and maxilla, therefore, we will still have to edit the images and make sure that all the pixels that belong to the mandible are really in the mandible mask.

![Mandible maxilla masks](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/0300029F_374x214.png)

Scroll through the coronal images and check if every pixel that belongs to the mandible is in the proper mask. Do you notice at position 64.50 that some pixels (at the left side in the image) from the maxilla are wrongly put in the mask of the mandible?

![Coronal wrong pixels](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002A0_448x277.png)

Move both indicators until their point of intersection indicates the wrong pixels (figure above). It concerns two layers of pixels, belonging to a tooth of the maxilla. In the two corresponding axial images (position -6,50 and -5,50), erase the tooth from the mask of the mandible. You cannot be mistaken, because that tooth is also indicated with the point of intersection of the indicators (figure below). If the two layers of pixels are shown in gray values in the coronal image, you can be sure you erased the whole tooth from the mandible mask. If not, move the indicators again in the coronal image so their intersection points to the wrongly colored pixels. In the axial image, remove the pixels that are indicated by the indicators from the active mask.

> **Note:** you can still access the 1-click navigation function by pressing the SHIFT button while you are editing. You can then click with your left mouse button on the point you want to navigate to.

![Axial erase left](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002A1_187x187.png)  ![Axial erase right](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002A2_187x187.png)

*Axial images (position: left -5,50, right -6,50): the indicators point out the tooth that does not belong to the mandible mask.*

In the sagittal image at position 87.25 (or the coronal image at position 43.25) another collection of badly masked pixels is visible. But now it's the opposite situation! Three layers of pixels that belong to the mandible are not in the mandible mask. Two layers belong to the maxilla mask and the other layer is the one we erased in the beginning to make the disconnection. Again, mark these pixels with the indicators as it is done in the figure below. In the corresponding axial images (at positions -4.50 and -3.50 and -2.50) the pixels (of a tooth) should be added to the mandible mask. We will make use of a local threshold to do this. To make this threshold clear, a short intermezzo is inserted below.

![Sagittal missing pixels](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002A3_334x213.png)

![Sagittal indicators](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002A4_448x270.png)

*Indicators point to pixels that should belong to the mandible mask (sagittal view).*

**Local threshold:** In the obturator case it is mentioned that there are three modes to choose from in the Edit toolbar i.e. draw, erase, and threshold. The threshold mode (Ctrl + T) is used to set a local threshold. This means that if you apply a local threshold in a particular area of one image, this threshold doesn't apply to the other images in the project. Remark that the threshold we've set in the beginning of this case was global and it applied to every image in the dataset.

When you activate this mode, the box with the two default threshold values is shown on your screen. To set a different local threshold, press one of the two arrow buttons and double click on a threshold value. After you changed the value, press **Enter**. When you move the square over the image while pressing the left mouse button, every pixel that comes to lie within the square and has a threshold in the threshold range you just set, will be added to the active mask. On the other hand, all the pixels that already belonged to the active mask and that don't have a gray value within the range will be removed from the mask.

![Local threshold range box](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002A5.png)

*The local threshold range*

For the moment we don't have to change the threshold values, but it will be used later on in this case to remove artifacts out of the image.

Maybe you now wonder why we will add the pixels of the teeth that belong to the mandible with this local threshold method and not with the draw mode we will use in the obturator case. With the draw mode, can't you also add pixels to a mask? Yes, that's true, but there is a difference! With the draw mode you add every pixel you touch with your cursor. With the threshold mode you do the same, but there is one more condition before they are really added: their HU values must lie in the range shown in the box. In this case, it's much safer to add pixels by taking into account their gray values. Our segmentation will be more accurate.

Press Ctrl + T. The Edit toolbar shows up and the threshold mode is already selected. Choose a circle as type of cursor and make it more or less the same size as a tooth. Make sure that the mandible mask is the active mask. Press the left mouse button and go over the tooth with your cursor. Make sure you got the tooth completely. You can check this very easily by looking at the sagittal or coronal image: if the wrongly masked layers now have the color of the mandible mask it's alright, otherwise you've forgotten some pixels. Suppose you added too much pixels, just press E (or select the Erase mode with your mouse) and erase them. If you repeat the thresholding in the necessary axial images (see before to know their positions) you should have a sagittal image like in the figure below. Now we can say that the whole mandible is in the mandible mask.

![Sagittal after local threshold](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002A6_448x271.png)

*Sagittal image after local thresholding*

##### Artifacts

The images still don't look nice, because of all the artifacts. We are going to get rid of them by again performing a local threshold, but not the default one like we just used to add the pixels.

To enter the Edit mode, press Ctrl + T. The threshold mode is already selected. Click on the top arrow of the threshold range box and double click the threshold 1 value. Change this value to 3000 (if you are working in Hounsfield Units) and press **Enter**. Because the Hounsfield Units of the artifacts are lower than the ones of the teeth. Go with your cursor over the artifacts and notice that they disappear. Why do we use this high local threshold? Because the HU values of the artifacts are lower than the ones of the teeth. So by setting a very high threshold the artifacts will be removed from the mask because their gray values are not in the range. Moreover, if you accidentally go with your cursor over the teeth, their pixels will remain in the mask, except for the edges (their HU are lower). If you removed the edges from the mask, don't panic. Set the threshold range back to the default one by clicking once on the lowest arrow and move your cursor over the tooth again to restore the edges. So, this is the way you should work. Scroll through the axial images and remove all the artifacts from the mask of the mandible.

![Artifacts left](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002A7_214x177.png)  ![Artifacts right](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002A8_215x177.png)

*The artifacts in the left image are removed with a local threshold. The right image shows the result*

##### Multiple particles

Let's calculate the 3D image of the mandible. Press the **Calculate 3D** button and select the mandible mask to be calculated (choose low quality). You get the message that the mask consists out of multiple parts. Answer "Yes".

![Multiple parts message](Resources/Images/Editing-Thresholding.png)

Visualize the 3D by pressing the **3D** button. Rotate the model and remark that there are little particles floating around the mandible due to which you got the message about the multiple parts. The particles are due to the editing you've done to remove the artifacts. To avoid this you have to do a region growing before calculating the 3D. Press the 3D view button again to get back the sagittal image. Press the **Region grow** button and click into the mandible. Change the name of this new mask to "Total mandible". Now calculate and visualize the 3D model of the final mandible. You can delete the first 3D (with the particles) listed in the 3D tab of the Project Management.

##### Scan prosthesis

Can you distinguish between the natural teeth and the scan prostheses in the 3D model of the mandible? It's quite simple; the natural teeth are connected to the bone, while the scan prostheses are not. There are 3 teeth of the scan prostheses at the patient's left side and one at his right side. In the figure below, the scan prosthesis (axial view) is marked with rectangles.

![Scan prosthesis axial](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002AA_315x226.png)

*Axial image indicating the prosthesis in the boxes.*

We would like to have the mandible without the prosthesis and the prosthesis itself into two different masks. There are two ways to achieve this. The first one is to proceed with the segmentation of the final mandible and to remove the prosthesis from the active mask. The second option is to perform a segmentation of the prosthesis. We opt for the latter. We will do a region growing of the prosthesis twice, once at either side. But, we first have to make sure that the prosthesis is completely disconnected from the natural teeth. The intention is to remove (from the final mandible mask) the pixels surrounding the prosthesis and the pixels connected to the prosthesis. The goal is to get the prosthesis nicely isolated in every image. Keep the following advice into account: remove enough pixels in the surrounding of the prosthesis, because sometimes in 2D it looks like there is no connection, but there is still one in 3D. So a 3D model can be very tricky!

![Project Management Masks tab](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002AB.png)

*Project Management – Masks tab*

Make the mask of the final mandible active and press the **Duplicate** ![Duplicate](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/020002AC_20x22.jpg) button in the Masks tab of the Project Management window. This way a backup mask is created that we can use to do the segmentation of the prosthesis, while the original Final mandible mask is left unchanged. The original one will be used later on to perform Boolean operations. Proceed with this backup mask (if you don't like the color, press the **Color** button in the masks tab and choose the color you like). Scroll through the axial images and remove (enough!) pixels surrounding the prosthesis from the active mask. In the figure below it is shown for the axial image at position -11,50.

If you think you disconnected the prosthesis completely, press the **Region Growing** button. Make sure your target mask is a new mask (if not, select "new mask" from the drop down list) and that you activate the **Leave Original Mask** option. This last option is very important!

![Prosthesis disconnected layer](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002AD_313x217.png)

*The prosthesis is disconnected in this layer.*

Click on the left or the right prosthesis. If you disconnected the prosthesis entirely, only the prosthesis should be shown in the color of the target mask. If this is not the case, make the previous mask active again, delete the last mask in the list (generated for the region growing) and remove more surrounding pixels from the backup mask. Also in the layers where you don't see the prosthesis it can be useful to remove some pixels belonging to the teeth next to the prosthesis. Repeat these actions for the prosthesis at the other side. Give the masks of both prostheses proper names.

#### Boolean Operations

Let's examine what we've obtained so far: the final mandible (with prosthesis), the left prosthesis and the right prosthesis in three different masks. That's nice, but we said earlier that we would like to have the mandible without prosthesis. We can achieve this with some Boolean operations. Press the Boolean operations button ![Boolean](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/020002AE_25x25.jpg). Let's use the following calculation:

Mandible without prosthesis = total mandible – left prosthesis – right prosthesis

Follow the steps below:

- **Mask A:** final mandible
- **Operation:** minus
- **Mask B:** left prosthesis
- **Result:** new mask (called mask C for reference)

After these options are set, press the Apply button.

- **Mask A:** mask C (obtained in the first step)
- **Operation:** minus
- **Mask B:** right prosthesis
- **Result:** new mask

After these options are set, press the **Apply** button.

Press the **Close** button. The last mask (the one that should be active now) contains the pixels of the mandible without the prosthesis. Calculate and view the 3D of this mask. Show also the left and right prosthesis. The other 3Ds can be set invisible. You should have a model that looks like this one.

### View of end result

![Simon final result](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002AF_382x305.png)

---

## Smart Expand

In the Smart Expand tutorial we will explain how Smart Expand can be applied to segment a liver. The Smart expand tool dilates a rough initial mask until it meets gray value gradients in the images and limit itself to the gradients. In other words, it stops expanding when it finds an edge in the image. The region of operation can be limited by setting the maximum expand distance.

## Opening the project

Open the project Liver.mcs. The project is a CT scan of the liver.

## Creating the source mask

The smart expand tool needs a source mask from which it grows. The growth is limited to the boundaries or gray value gradients in the images.

To create a source mask, first create an empty mask. This can be done by clicking on the New mask button in Project Management tab.

![New mask](Resources/Images/image28.png)

When you click the New button, the threshold window will open. The threshold will not be used during the segmentation process, but it is used when calculating the 3D model. A good selected threshold will result in a better looking 3D. In this case we will use a minimum threshold of -50 and a maximum threshold of 280.

![Threshold window](Resources/Images/image29.png)

Next, select the just created mask and click on "Clear Mask" button in the project management, see below. This will clear the mask you have selected. Now you have an empty mask and we will edit this to create the input for the Smart Expand tool.

![Clear Mask](Resources/Images/image30.png)

## Initialize the source mask

To initialize the source mask, we will roughly indicate the source mask on several slices. When drawing the mask it is important to only select the liver and not any neighboring anatomy.

Go to the Edit masks tool in segmentation toolbar and select the Draw option (Alternatively, press Control on your keyboard and hold while pressing D to go to this option directly.) We will now go to the Full screen mode on the Axial slices and start to draw in the interior of the liver as shown below. We should paint only the inside of the liver.

Start from Slice 100.00 and use Page down key to go 10 slices down. Repeat the process till slice 265.00.

![Draw inside liver](Resources/Images/image31.png)

The results should look somewhat like shown.

![Result slices 1](Resources/Images/image33.png)
![Result slices 2](Resources/Images/image34.png)
![Result slices 3](Resources/Images/image35.png)
![Result slices 4](Resources/Images/image36.png)
![Result slices 5](Resources/Images/image37.png)
![Result slices 6](Resources/Images/image38.png)

Next, we will repeat the process for Sagittal slices. We'll start at slice number 48.2383 and go 10 slices up like before using Page up. Repeat the process till slice 230.2695. The mask will look somewhat like this:

![Sagittal result 1](Resources/Images/image39_235x118.png)
![Sagittal result 2](Resources/Images/image40_235x118.png)
![Sagittal result 3](Resources/Images/image41.png)
![Sagittal result 4](Resources/Images/image42_235x118.png)
![Sagittal result 5](Resources/Images/image43_235x118.png)
![Sagittal result 6](Resources/Images/image44.png)

Once complete, look at the coronal view and you will see a grid like mask.

![Coronal grid](Resources/Images/image45.png)

## Launch Smart Expand

Launch the Smart Expand tool from the Segmentation toolbar by pressing the ![Smart Expand](Resources/Images/Smart_Expand.png) icon.

In the Source Mask field, choose the mask you just created and confirm target mask field is set to New Mask. Since we made slices with maximum distance of 10 slices, we can set the Maximum Expand Distance field to be 10 pixels. Click Apply and let the algorithm run.

![Smart Expand dialog](Resources/Images/image47.png)

After the algorithm finishes, the result should be a full mask.

![Full mask result](Resources/Images/image48.png)

Run the Smart Expand tool again, to obtain a better result.

![Better result](Resources/Images/image49.png)

## Smooth mask

The result of Smart Expand will typically show spikes. Click on Smooth Mask in the segmentation toolbar to obtain a smoother result. By clicking multiple times on Smooth Mask you will smooth more.

## Calculate 3D

Click on the Calculate 3D button.

![Calculate 3D](Resources/Images/Calculate3D_common.png)

The Calculate 3D Dialog box is displayed. Here select Custom and click on options. In the custom dialog select triangle reduction and smoothing.

![Custom options](Resources/Images/image55_463x427.png)

Click OK, followed by clicking calculate. A 3D model of the liver will be visualized in the 3D view.

![Liver 3D](Resources/Images/image53_375x267.png)

---

## Hip

In this tutorial we will discuss some of the possibilities of the Analysis module. To finish this tutorial you need to have a license for the Analysis module.

The topics that will be discussed are:

- Opening the Project
- Preparation of the data
    - Thresholding
    - Region growing
- Calculation of the Polylines
- Patching of the contours
- Creation of Analysis objects
- Visualization possibilities

### Opening the project

The objective for this part is the creation of a file ready to use in all CAD-systems supporting the IGES-interface. The part of the "Hip" we'll focus on is the right femur of the patient (left in the images). In this IGES-file a basic reference system calculated on the data as well as a partial modeling of the outer contours using freeform surfaces will be present.

It is strongly advised to first follow the tutorial **Simon** to obtain the necessary skills for segmentation and image processing.

In the **File** menu, select **Open** (Ctrl+O). The Open dialog box shows all projects in the working directory. Double click the Hip.mcs file (Mimics project file).

#### Thresholding

A good minimum threshold value for this case is 1235 (Gray Values) or 211 (Hounsfield values). Set this threshold in your base mask and apply it. The procedure is explained in detail in **Simon**.

#### Region growing

We want to make a model of the right femur (left in the image set). Therefore use the following steps:

- Click the **Region Growing** button or press Ctrl + R.
- Set the **Source** to Green (if this is your base mask) and **Target** to New Mask. Check the **Multiple layer** box.
- Click the left mouse button on one point of the right femur (left on the images). The right femur has now been grown into a new mask (Normally if you have started fresh the femur will be in the yellow mask now).

To calculate your 3D, go to Project Management, Masks tab, select the yellow mask and press the **Calculate 3D** button. The yellow mask will be automatically selected in the Calculate 3D window, but you need to set the **Quality** to High and press the **Calculate** button.

You can find more details about this in **Simon**.

### Calculation of the Polylines

Go to the Project Management.

![PM Masks tab Hip](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002B0_236x217.png)

*Project Management - Masks tab*

Select the yellow mask and click on the action button, select the **Calculate Polyline** option from the action list. The Create Polylines dialog box appears with the Yellow mask already checked; click **OK**. The borders of your yellow mask will be calculated and displayed as a polyline in both 2D and 3D images.

![Polylines 3D](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002B1_207x182.png)

*3D view of the polylines*

You can also calculate polylines by clicking the ![Polyline](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/020002B2_25x25.jpg) button in the segmentation toolbar.

### Patching of contours

Since we are only interested in the outer contours, we need to select these out and grow them to a new set of polylines.

Go to layer -523 and zoom in on the right femur in the 2D image (xy plane).

Click the Polyline Growing button ![Polyline Growing](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/020002B3_25x25.jpg) in the Analyze toolbar.

Set all parameters as displayed in the image below: i.e. the set to start from, the set that will contain the grown polylines. In order to select a polyline, you need to draw a rectangle over it or simply click on its contour. Hold the left mouse button down, drag it and then release the left mouse button.

![Polyline Growing dialog](Resources/Images/Polyline_Growing.png)

| Before                                                       | After                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![Before polyline](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002B5_180x155.png) | ![After polyline](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002B6_179x154.png) |

The growing of the polylines stopped at layer -513 because of a small extension on the bone. This needs to be removed in layers -513 and -511. Afterwards, the polylines need to be updated and then we can proceed with the polyline growing:

- Click the **Edit masks** button and go to the **Erase** mode or press Ctrl + E
- Make sure that the Yellow mask is Active
- Erase the extension on the bone ![Extension](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002B7.png)
- Press the Ctrl + U key or the **Update Polylines** button in the Edit toolbar
- Repeat this for the following images.

Scroll back to image -513 and click the **Grow Polylines** button. Set "selection 2" as the target polyline and use 96 % as matching parameter. Select the polyline.

Scroll to image -485 (figure below).

![Image -485](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002B8_448x384.png)

*Image -485 of the Hip*

At this slice you see a cavity in the contour. If you want to restore this with editing, keep in mind that it will be the yellow contours that will be updated, so we need to remove the pink polyline first.

Do a Polyline Growing from Selection 1 to a New Set; be sure to turn Auto Multi-Select off. You can delete this set by selecting it in the Project management and then pressing the **Delete** button.

Lose the cavity by drawing in the mask and updating the polyline (Ctrl + U).

Similar editing and updating of the polylines needs to be done on slices: -483 till -479, -475, -471 (on the femur head). Don't forget to update for every image.

When all corrections have been made, the polyline growing can continue.

Go back to layer -485 and perform the Polyline Growing (from Set 1 to Selection 2, matching parameter 95 %, Auto Multi-Select on)

Once all the editing is performed properly, all layers until -477 will be stored in Selection 2.

The femur head and the greater trochanter will be grown into new selection sets. The end result should look like the figure below.

![Polyline sets](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002B9_273x278.png)

*Polyline sets*

### Creation of Analysis objects

On the great trochanter and on the lower part of the femur we will fit a Free Form Surface, on the femur head, we will fit a sphere.

In the Project Management on the Polylines tab, you will find a button **Fit Surface**. Choose Selection 2 and press the **Fit Surface** button. The following dialog box will appear.

![Surface Fit Parameters](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002BA_323x463.png)

*Surface Fit Parameters*

You can accept these default values and a Free Form Surface will be fitted on Selection 2.

> **Note:** Some caution in increasing the number of control points is advised. The basis of a B-spline is a polynomial and a polynomial has the tendency to wave. So, if the number of points is too high, the fit on the polyline will become worse.

Repeat this set on Selection 4.

The Free Form Surfaces are visible in 3D as a shaded surface and in 2D you will see a cross-section on every layer of this Free Form Surface.

To fit a Sphere on Selection 3, go to the **Analyze** menu and select **Sphere > Fit on Polylines**. Choose the correct polyline set.

The result of all these fittings should look like following figures:

| Fitted objects                                               | Imported STL                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![Fitted objects](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002BB_207x187.png) | ![Imported STL](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002BC_207x173.png) |

### Visualization possibilities

When a prosthesis is designed, the STL file can be loaded in Mimics. One can rotate and move this prosthesis to obtain the best fit of the prosthesis onto the femur and check the design related to the bone structures.

You can import an STL file in the Project Management from STLs tab. Click the **Load STL...** button. Browse to the MedData folder and select the prosthesis.stl. The STL file will be visible both in 2D (as cross-sections) and in 3D.

To adjust the position of the STL file, click the **Move** button to move the STL file or the **Rotate** button to rotate it. Both actions can be performed in 2D as well as in 3D.

---

## Obturator

In the previous cases we have segmented bone structures, whereas in this project we are going to make a soft tissue model. An interesting application is the modeling of the soft tissue around the cavity of the mouth. Such a model can be used as a mold for obturator prostheses. In the case study following this introduction we will do just that.

How are we going to model this soft tissue? Since we are only interested in the area around the cavity, we need to limit the model to the region of interest. By erasing one layer from the active mask in every direction, the cavity and the soft tissue around it will be separated from the rest of the image. This way the region of interest is captured in a 3D box delimited by the removed layers. Next we perform a region growing that starts in the region of interest. Because this region is separated from the active mask, only this area will be put into a new mask after the region growing is done. From the new mask a 3D model can be calculated which will contain just the cavity of the mouth and the soft tissue surrounding it.

The topics that will be discussed in this tutorial are:

- Case Study
- Preparation of the data
- Windowing
- Orientation
- Thresholding
- Editing
- Region growing
- View of the end result

#### Obturator prosthesis for oncologic patients

Case presented by Dr. L.L. Visch from Daniel den Hoed Kliniek Rotterdam.

The first picture shows the cavity in the mouth of the patient after resection of a tumor. In order to protect the tissue weakened by irradiation and to be able to breathe and eat normally, this hole needs to be filled by an implant.

![Cavity](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002BD_320x212.png)

A CT-scan of the patient was made. The soft tissue around the cavity, clearly visible on the scans, was modeled. This model served as a direct mold for the implant.

![CT scan](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002BE_322x212.png)

The implant, called an obturator prosthesis, was cast from the mold in a bio-compatible silicone.

![Implant](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002BF_322x228.png)

Absolutely no surgery was needed to implant the obturator prosthesis. As the silicone prosthesis is plastic deformable, it can be implanted very easily.

The prosthesis fits the cavity much better than ever could have been achieved by using conventional impression techniques. These traditional techniques produce a master of the obturator prosthesis by making an impression of the cavity in a deformable plastic material.

![Traditional impression](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002C0_322x218.png)

The prostheses cast from such masters are always less accurate because of the presence of undercuts (the impression technique is not sensitive to local internal broadening of the cavity) and can severely damage the sensitive and vulnerable surrounding tissue.

The soft prosthesis is fixed by means of magnets on a hard dental implant. This makes it possible to take it out for inspection and to replace it afterwards.

#### Preparation of the data

In the **File** menu, select **Open** (ctrl+O) or click the button ![Open](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/020002C1_25x25.jpg). Double click the obturator.mcs project.

#### Windowing

For correct windowing see the windowing procedures in "**Mimi**".

#### Orientation

When the project is loaded, a Change Orientation window pops up.

In the axial image you see the orientation strings L and R, which stand for Left and Right respectively. In the coronal and sagittal image several Xs are displayed instead of the orientation strings. Move the mouse cursor to the top X in the sagittal or coronal image. The cursor is changed to a hand and when you right-click, a menu appears with all possible orientation strings. Select "Top". Remark that all other orientation strings are completed automatically.

Do the same to set the Anterior-Posterior orientation parameter looking at the image displayed.

![Orientation X](Resources/Images/Images2/image55.png)

You can always change your orientation parameters, going to **Image > Change Orientation**.

#### Thresholding

A reliable way to define an appropriate threshold is to make use of a profile line (see "**Mimi**"). Press the **Profile line** button ![Profile line](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/020002C3_25x25.jpg) and draw a line in the axial image over the cavity.

![Profile line over cavity](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002C4_448x347.png)

*Profile line over the cavity of the mouth*

See the figure above (axial image on position 374) to have an idea where to place the profile line. You get a profile like shown in the image below. You can clearly see the transition from the soft tissue to the cavity. Press the **Start thresholding** button. To visualize all the soft tissue in the mask, drag the lowest threshold line to the value -44 (Hounsfield scale). Press again the **End thresholding** button and answer "Yes" to the question whether you want to save the threshold value or not. Close the window.

![Profile line dialog Obturator](Resources/Images/Profile_Line_Obturator.png)

### Editing

In the axial image, go to position 387.00. Press the **Edit masks** button ![Edit masks](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/020002C6_25x25.jpg). The edit toolbar is displayed on your screen. Your cursor has become a little square. If not, go to Type and select a square from the drop down list. Notice that the length and the width of the square are displayed and can also be altered. The easiest way to change the size is to press the control key and your left mouse button simultaneously and to move to the right/left to make the square bigger/smaller.

![Edit toolbar Obturator](Resources/Images/EditMask_obturator_621x77.png)

The three modes available are listed below. To make a mode active, just click in the little circle on the left of the mode or press the first letter of the desired mode. When the edit mode is not yet selected and you use the shortcuts between parentheses below, the edit toolbar appears and the associated mode is activated.

- **Draw** (Ctrl + D): Every pixel that lies within the shape of your cursor, while pressing the left mouse button, will get the color of your active mask. In other words, you add pixels to the active mask by going over the pixels with the square.
- **Erase** (Ctrl + E): This mode is the opposite of the draw mode. You remove all the pixels from the active mask by moving the square (keeping the left mouse button pressed) over the pixels in the image.
- **Threshold** (Ctrl + T): This mode is used to set a local threshold. This means that if you apply a local threshold in a particular area of one image, this threshold doesn't apply to other images in the project. Remark that the threshold we've set in the beginning of this case was global and it applied to every image in the dataset.

When you activate this mode, a box with the two default threshold values is displayed on your screen. To set a local threshold, press one of the two arrow buttons and double click on a threshold value. After you have changed the value, press Enter. When moving the square over the image while pressing the left mouse button, every pixel that comes to lie within the square and has a threshold within the threshold range you set, will be added to the active mask. On the other hand, all the pixels that were already part of the active mask and that don't have a gray value within the range will be removed from that mask.

![Local threshold box](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002C8.png)

For the current case we are only going to use the draw and the erase mode. In the Simon case we already illustrated the threshold mode.

Working on the axial image in position 387 activate the Erase mode (Ctrl + E) and set a very large square (for example, 200 by 200). Press your left mouse button and wipe off all the color in the image. Be sure not to forget any pixels! Close the Edit toolbar.

![Erase before](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002C9_211x187.png)  ![Erase after](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002CA_206x187.png)

Notice in the sagittal image that one layer is shown in gray values. In the figure below, the sagittal image is displayed and the arrow points to the layer that has been removed from the active mask (the slice indicator is moved down to see this).

![Sagittal erased layer](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002CB_352x224.png)

To see the result of erasing the mask in one layer, we will now perform a region growing ![Region grow](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/0300029B_22x22.png). Select the axial image at a position lower than 387.00 (= the position of the image we removed from the active mask). Press the Region Growing button, a window will be displayed on the screen.

![Region Growing Obturator](Resources/Images/RegionGrowing_obturator.png)

Check both the Multiple Layer and Leave Original Mask checkboxes and click on an arbitrary position in the active mask. You see that all the images at a position lower than 387.00 are put into a new mask (yellow mask in figure below). Close the region growing toolbar.

![New mask after region grow](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002CD_364x232.png)

Why are the images above this position not included into the new mask? As you already know, a region growing looks for pixels that are connected to each other and puts them into a new mask. But, because we have disconnected the lower images from the higher ones, we have limited the area of the region growing. This will be the trick we will use to get our region of interest into a separate mask.

How will we proceed? In the same way as above, we are going to erase a complete layer from the active mask on every side so that our region of interest is completely surrounded by these removed slices. After we perform a region growing within that region, we should have the oral cavity and the surrounding tissue in one mask, like we wanted.

Activate the axial image. Go to position 362.00 and press Ctrl +E (or press the **Edit masks** button and select the **Erase** mode). Make a big square and erase all the pixels from the active mask. Take a look at the sagittal image. Two horizontal lines are shown in gray values. The top and the bottom of our box are now defined.

To set the left and right boundaries of the box, you have to remove two layers from the mask in the sagittal image. Try to visualize the situation and make sure you understand why we will now operate in the sagittal image. Erase all pixels from the active mask at position 126.49 (left boundary) and 42.05 (right boundary) in the sagittal image. In the axial image two vertical lines in gray values are visible.

To close our box, a separation still has to be made on the posterior side. Activate the coronal image and remove all pixels from the active mask at position 76.61. In the axial image the removed layer is visible. Setting a boundary on the anterior side is not necessary. In figure 5-10 you can see the boundaries of the mouth cavity on the yellow mask.

### Region growing

Now that the box is delimited by the layers removed from the active mask, a region growing can be performed to get the obturator into a new mask. Go to an axial image that has a position between 362.00 and 387.00. This is to make sure that the starting pixel for the region growing lies within the region of interest. Press the **Region grow** button and click in the axial image within the box.

![Boundaries axial](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002CE_300x232.png)

*Boundaries of the mouth cavity in the axial image.*

The mouth cavity is now within a new mask. In the figure below you clearly see the mouth cavity within the active (blue) mask from the axial and the sagittal viewpoint.

![Axial and sagittal cavity](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002CF_300x232.png)  ![Sagittal cavity](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002D0_364x232.png)

*Axial and sagittal view of the mouth cavity*

Because we disconnected the pixels of the mouth cavity from the other pixels in the original mask, the region growing was confined to the region of interest.

Press the **Calculate Part** button ![Calculate Part](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/020002D1_25x25.jpg) and select the mask of the mouth cavity. Choose custom quality and press the **Calculate** button. The processing of the 3D model is started.

On the right of the Part you see a toolbar and a button where you can select some predefined viewpoints for your 3D model ![Viewpoints](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/020002D2_25x25.jpg). If you press the bottom view you should obtain a model as shown in the figure below. You can also enable transparency using the ![Transparency](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/020002D3_25x25.jpg) button.

### View of end result

![Obturator result](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002D4_318x280.png)

---

## Import Raw Images

In this case we will show you how you can use the manual import function to import any image data you want. The topics that will be discussed in this tutorial are:

- Raw Import
- Edit images

#### Import images

Select **New Project Wizard** from the **File** menu. In the New Project Wizard select all the images at "MedData\DemoFiles\RAW_Images" directory and click **Next**.

![Raw import file selection](Resources/Images/Raw_Import_554x438.png)

If the images are in Raw format, the New Project Wizard will automatically take you to the following steps. You can also force raw import by selecting the **Raw import** option from the dropdown list at the bottom right of the dialog. If this option is selected, Mimics will import all images as RAW images. When you press the **Next** button, you will see that the files are recognized as "unknown files" in the "Import log" window. Click **Next** to go to the Raw image properties window.

![Raw image properties](Resources/Images/raw_import_tutorial_491x356.png)

In the Raw image properties window you will have to enter the parameters of the scan, namely, the Scan resolution, the Image parameters and the Pixel properties. This information is usually provided along with the scan by the radiologist. For this case, the Scan resolution is 0.5 X 0.5 X 1 mm and the Image parameters are 128 X 128 pixels. The pixel values are in Signed Long format with Low byte order Byte swapping. When you have entered the correct parameters, you can preview the images and **Next** button will be activated.

Following is some more explanation on the parameters.

##### Scan resolution

Here the sizes of the pixels have to be entered. For this example, each pixel are 0.5mm in X-direction, 0.5mm in Y-direction and 1mm in Z-direction. If the image slices are taken axially, then Z-direction would be equivalent to slice distance.

##### Image parameters

The file header size is calculated automatically, based on the file size, the resolution of the images and the pixel type.

Typically a file contains both a file header and the image itself (in some rare cases also a footer is present). The file header can contain information about pixel size, patient data, etc. The image is a matrix of pixels. The horizontal (or vertical) image size is equal to the number of pixels in that direction.

![Image parameters example](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002D7_317x313.png)

The number of pixels in vertical and horizontal section is the height and the width of the images. Common sizes of images are: 256 * 256, 512 * 512 and 1024 * 1024. In this example, the images have a resolution of 256*256.

##### Pixel properties

The number of bytes per pixel depends on the type of the pixel. Some examples of pixel types and their respective sizes (note that these types can be either signed or unsigned, however, this does not affect their size):

- Byte: 1 byte
- Short: 2 bytes
- Long: 4 bytes
- Float: 4 bytes

If you fill these values in, you will see that Mimics will set the file header size to 8432 bytes.

Byte swapping determines the order in which the images are read. You can try different options for byte swapping parameter and preview the images. For this case, when **High byte first** is chose, there are local distortions all over the image, because the data is read in the wrong order.

For this example, the pixel type is Signed Short and Low Byte First for the parameter Byte Swapping.

##### Study information

Here you may fill in an appropriate name for the patient name. This will be the name that is used for your project.

### Edit images

If the images look good in the preview, click the **Next** button in the Raw image properties window. In the Edit images window, you may crop or resample the images.

![Edit images window](Resources/Images/raw_import_tutorial2_513x372.png)

Click on the Pixel Mapping tab to view the histogram of the pixels. Here you can also map the pixel gray values to a custom range by moving the sliders from the ends of the histogram. For this case, the imported pixel gray values will be mapped to a 16 bit gray value range, as shown here.

![Pixel mapping](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002D9_554x343.png)

Click **Next** and then you will see the familiar Check orientation window, where you can set the orientation into the Mimics project.

---

## Simulation

In the Simulation Tutorial we will explain some of the functions that are available in the Simulation module. We will start with a dataset of a skull with a hole in it and explain how to do the segmentation, how to calculate the 3D, how to cut, split and reposition a custom implant. The Simulation module has to be licensed to be able to conclude this tutorial.

The topics that will be discussed in this tutorial are:

- Opening the Project
- Windowing
- Thresholding
- Region Growing
- Calculating a 3D
- Cutting
- Splitting
- Mirroring
- Repositioning

### Opening the project

In the **File** menu, select **Open** (Ctrl+O). Browse to the directory where you have installed the extra Tutorial Files and double click the Skull_with_hole.mcs file.

### Windowing

For correct windowing see the windowing procedures in "**Mimi**".

### Thresholding

Go to an axial image where the skull is visible. Press the **Profile line** button and draw a line over the bone. The figure below shows a profile line and the corresponding profile dialog box. Press **Start thresholding** and drag the threshold line to a value of about 1250 (Gray value scale). End the thresholding and save your settings. Close the dialog box.

![Threshold profile skull](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002DA.png)

![Profile line dialog Simulation](Resources/Images/Thresholding_Simulation_602x494.png)

*Profile line over the bone (upper image) and the corresponding profile dialog box*

### Region Growing

Now we will use the region growing tool to separate the skull from the artifacts and noise in the images:

- Click the **Region Growing** button or press Ctrl + R.
- Set the **Source** to Green (if this is your base mask) and **Target** to New Mask. Check the Multiple layer box.

![Region Growing Simulation](Resources/Images/RegionGrowing_obturator_535x71.png)

- Click the left mouse button on one point of the skull. The skull has now been grown into a new mask.

![Skull mask](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002DC_401x361.png)

### Calculating a 3D

Go to the Project Management by clicking its icon ![PM icon](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/0200029C_25x25.jpg) and choose the Masks tab.

You'll see all created masks listed with their respective threshold. Selecting one mask will make it active and it will appear in the Active Mask field in the visualization toolbar automatically. It is possible to hide/show a mask by clicking on the glasses.

![Masks PM Tab](Resources/Images/Masks_PMTab.png)

Click on the **Calculate Part** button.

![Calculate Part](Resources/Images/Calculate3D_2.png)

The **Calculate 3D** Dialog box is displayed. Here you can mark (with a green dot in the column called "Selected") which masks you want to visualize and calculate the 3D by clicking on the **Calculate** button.

Select the "Skull" mask if it is not already selected and click on the **Calculate** button.

### Cutting

After the calculation of the 3D you will see a 3D representation of the Skull mask. To be able to make a cut that fits well, make the skull transparent by clicking on the ![Transparency](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/020002DF_25x25.jpg) button and choose to view the skull from the Right view. Now you can pan and zoom so you can see the hole clearly.

| Skull transparent                                            | Skull zoomed                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![Skull transparent](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002E0_213x179.png) | ![Skull zoomed](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002E1_214x180.png) |

If you then zoom and pan, you can clearly view the hole in the skull through the intact side.

![Hole in skull](Resources/Images/skull_with_hole 3d_simulation_460x282.bmp)

This way we can easily draw around this hole. To do this use *Cut with Polyplane* option, go to **3D Tools > Cut > With Polyplane** in the menu. You will see following dialog:

![Cut with Polyplane dialog](Resources/Images/Cut_with_polyplane_simulation.png)

Select the 3D from the skull in the **Objects to Cut** list. The **New** button is already enabled so we can immediately start drawing a cutting path. Do this by clicking several times with your left mouse button around the hole like below. To end the drawing, double click with your left mouse button.

![Drawing cut path](Resources/Images/skull_with_hole 3d_simulation2_576x352.bmp)

You can see that a cutting path has been added to the cutting path list. You can now make the 3D opaque again by clicking on the ![Transparency](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/020002DF_25x25.jpg) button. You can then rotate the 3D to determine if the cut went through the whole skull or not:

![Check cut depth](Resources/Images/skull_with_hole 3d_simulation3_578x542.bmp)

As you can see, it would be best if we adjust the depth of the cutting path. You can do this by clicking on the **Properties** button while the cutting path is selected. This will open the cutting path properties dialog:

![Cutting path properties](Resources/Images/Cutting_plane_properties_simulation_332x266.png)

Adjust the Depth of the cutting path from 20.0mm to 30.0mm and enable the **Closed** checkbox (this will close the cutting path). Click on **Preview** to view the result. When you are happy with the result, close the Cutting Path Properties by clicking on the **OK** button. Enable the **Keep Originals** checkbox (since we want to keep the original 3D) and finish the cut by clicking on the **OK** button of the **Cut with Polyplane** tool.

You can see in the 3D objects list that a new 3D object was added.

### Splitting

The next step is to split the two cut parts of the newly generated 3D. To do this, go to **3D Tools > Split** in the menu.

![Split dialog](Resources/Images/Split_simulation_461x193.png)

Select the freeform object, choose to keep all parts and disable the **Keep Originals** checkbox. You can then click on **Preview** to preview the split and then on **OK** to apply the split.

![Split preview](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002E8_448x383.png)

As you can see, two different objects were created and have been given a different color.

You can make the largest part invisible since we will only need the small part to fix the defect in the skull.

### Mirroring

To mirror the part to the other side of the 3D, we will need a mirror plane. The Mimics simulation module generates a default sagittal plane, but we will have to adjust this plane a bit to make sure it's suitable for this dataset.

To do this, go to the **Simulation Layout** (by pressing F5 or by going to the **View** menu, choose **Layouts** and then **Simulation Layout**). Then make the original skull visible and go to **Simulate > Measure and Analyse** in the menu.

![Measure and Analyse](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002E9_630x297.png)

You can see in the right dialog that you can change the Sagittal Plane. Click on the **Change** button and adjust the Sagittal plane (by dragging the white points with your left mouse button) in the axial images to make sure the sagittal plane goes through the center of the nose.

![Adjust sagittal plane](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/020002EA_448x374.jpg)

After this, close the Measure and Analyse tool by going to the Simulation Menu and choosing Measure and Analyse again. Then mirror the part by going to the Simulation Menu, from the **3D Tools** select Mirror. Select the correct part and mirror plane and disable the **Keep Originals** checkbox and click on the **OK** button to apply the mirroring.

![Mirror dialog](Resources/Images/Mirror_simulation_513x165.png)

![Mirrored part](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/020002EC_448x327.jpg)

As you can see, the part is mirrored, but not correctly positioned. We will reposition this part in the next section of the help.

### Repositioning

To reposition the part, go to **Align > Reposition** in the menu. This will open following dialog:

![Reposition dialog](Resources/Images/Repositioning_simulation_628x87.png)

Select the Mirrored part and start the repositioning. The easiest way to do this, is to first reposition the part with the mouse and then do some fine-tuning with the parametric translation and rotation tools.

Click on the **Move with Mouse** button and reposition the part. You can translate the part by dragging the center point with your left mouse button and rotate the part by dragging the corners of the selection box with your left mouse button. Keep in mind that you can also reposition in the 2D views, this makes it easier to get a good fit. During repositioning it is also possible to scroll through the axial images to make sure the fit is optimal on all slices.

![Repositioning with mouse](Resources/Images/Repositioning_simulation2_622x381.png)

When you are happy with the fit, you can click on the **Analyze Motion** button to see the final translation and rotation of the part.

![Analyze Motion](Resources/Images/Analyze_motion_simulation.png)

To apply the reposition, click on the **OK** button. You can then export the part and the skull to STL files and continue working on the custom implant in your design software.

---

## FEA

In the FEA Tutorial we will explain the work-flow for making a FEA analysis on a model of the Femur. We will start with a dataset of a Femur and explain how to do the segmentation, how to calculate the Part, how to remesh the Part and how to assign materials to the Part. The FEA and STL+ module have to be licensed to be able to conclude this tutorial.

The topics that will be discussed in this tutorial are:

- Opening the Project
- Calculating a Part
- Remeshing the Part
- Creating the volume mesh based on the remeshed Part
- Material Assignment
- Exporting the Volumetric Mesh

### Opening the project

In the **File** menu, select **Open** (Ctrl+O). Browse to the directory where you have installed the extra Tutorial Files and double click the Femur.mcs file.

### Calculating a Part

There is already a Yellow2 mask available in this dataset that will be used to calculate a Part. In the **Calculate Part** dialog select the High quality setting and click on calculate.

![Calculate Part FEA](Resources/Images/fea_tutorial_calculate_part_358x267.png)

### Remeshing the Part

In this step the Part needs to be remeshed to be optimal for FEA purposes. You will notice that there are two Parts, select the Yellow 2 part. The FemurShaft model will be used in the non-manifold assembly tutorial. To export the Part to 3-matic, go to the **FEA -> Remesh** in the menu. This will bring up the following dialog:

![Remesh dialog](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/030002F2.png)

Select the Yellow 2 Part and click on **OK**.

To get the optimal result it is common to follow the next steps in 3-matic:

- Define shape parameters
- Inspect the quality of the surface mesh
- Reduce the details of the anatomical part
- Remesh the surface elements
- Generate the volume mesh
- Visualize volume elements
- Analyze Mesh quality

For details on how to get the optimal mesh for your Part see the **Chapter 5: Remesh** of the Tutorials in 3-matic. You can find it under **Help** -> **Tutorial...** in 3-matic.

When you are satisfied with the quality of the mesh copy your object by selecting your object and pressing Ctrl+C on your keyboard. Open your Mimics window and paste your object there by pressing Ctrl+V. The volume mesh will be available in the FEA mesh tab in the project management section. These meshes can then be exported to your FEA software.

### Material Assignment

When you have created a volumetric mesh from your remeshed object, you can perform the material assignment in Mimics. You can see the mesh listed in the FEA mesh tab.

![FEA mesh tab](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000310_236x224.png)

> **Note:** We will use gray values for this tutorial, so if you are working in Hounsfield units, please change this by going to the Edit menu, choose Preferences and change the Pixel Unit in the General tab.

With the FEA mesh of the Femur selected, click on the **Materials** button. Mimics will display a message that the gray values for this mesh have to be calculated before you can do a material assignment. Choose "Yes" to continue. After the calculation you will see following dialog box:

![Materials dialog](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000311_739x579.png)

Mimics shows for each gray value the amount of elements that were assigned that particular value. We will then convert this gray value to material properties. In this tutorial we will use the uniform method.

**STEP A:** If the Gray value based method is not selected, click on the radio button next to Uniform.

**STEP B:** Enter the number of materials in the edit box. We will use 10 materials for this tutorial. The FEA module will now divide the range of gray values that occur in the volume mesh into 10 equally sized intervals that each represents a material. You can see this discretization by choosing the Materials histogram. Select **Limit to Mask**: Green 2. The limit assignment to mask intercepts the deviation in the boundary elements due to the partial volume effect. As boundary voxels typically represent multiple tissues by excluding these voxels, the material assignment will become more accurate.

![Materials histogram](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000312.png)

**STEP C:** Enter a density expression to convert the gray value of each material to a density. For this tutorial we will use following expression: Density = -13.4 + 1017 * Gray value.

**STEP D:** Choose to write out only the Young's modulus material properties in the exported file by deselecting the selection boxes before Density and Poisson Coefficient. We will use following expression for the Young's modulus: E-Modulus = -388.8 + 5925 * Density.

**STEP E:** Check the values for the materials that will be assigned in the material editor:

![Material editor](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000313_697x559.png)

**STEP F:** Press the **Apply** button to assign the materials to the FEA mesh. The elements of the FEA mesh will be colored according to their materials:

![Colored FEA mesh](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000314_449x404.png)

This volumetric mesh can then be exported together with the material assignment (in this case only the E-Modulus).

> **Note:** It is also possible to use different expressions for different ranges or different masks material assignment. For more detail on this check the section on Material Assignment using Lookup Files.

### Exporting the Volumetric Mesh

The volumetric mesh, together with the material assignment can be exported to ANSYS, Patran Neutral, and Abaqus files and can then be used to do FEA analysis on the mesh. To export the mesh go to the **File** menu and choose **Export**. Then go to the FEA tab, add the correct mesh to the export list, choose the required format and export directory and click on the **OK** button.

![Export FEA mesh](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000315_414x388.png)

---

## CFD

For a CFD Tutorial see the **Chapter 5: Remesh** of the Tutorials in 3-matic. You can find it under **Help** -> **Tutorial...** in 3-matic.

In the CFD Tutorial a typical remesh workflow it is described:

- Importing the images
- Segmentation
- Calculation of a Part
- Remeshing the Part
    - Optimisation of the mesh
    - Materials assignment

---

## Non-Manifold Assembly

The non-manifold assembly Tutorial explains step by step how to obtain matching surfaces between the bone and an implant. First we will register the femoral head prosthesis on the Femur. Secondly we will use the cutting tools of the simulation module to perform an ostectomy of the femoral head. In 3-matic we will combine both the femur shaft and the implant to ensure perfectly coinciding nodes between them. The Simulation and FEA module have to be licensed to be able to conclude this tutorial. In case you do not have the simulation module you can skip the Ostectomy of the femoral head and still perform the remeshing part of the tutorial.

The topics that will be discussed in this tutorial are:

- Opening the Project
- Calculating a Part
- Registration of the implant
- Ostectomy of the femoral head
- Remeshing the femur and implant
- Creating a volume mesh
- Exporting the remeshed Parts

### Opening the project

In the **File** menu, select **Open** (Ctrl+O). Browse to the directory where you have installed the extra Tutorial Files and double click the Femur.mcs file.

### Calculating a Part

There is already a yellow mask available in this dataset that will be used to calculate a Part. Select the **Yellow** mask and click on the **Calculate Part** icon in the Masks toolbar. In the **Calculate Part** dialog select the **High** quality setting and click on **Calculate**.

#### Import the STL

Select STL by going to **File > Import > STL** in the menu. From the STL folder load the Implant.stl.

#### Point registration

The Point registration will be used to bring the implant nearer to the Femur. Indicate a start points on the STL and their corresponding end point on a 3D model or in the 2D views. Mimics will then calculate the transformation matrix that should be applied to have the best fit between the start and end points and applies the transformation matrix on the selected STLs.

Go to **Align > Point Registration** in the menu. Click on **Add point**, add a start point on the top of the implant head and put the corresponding end point on the femur head. Place a second set of points on the end of the implant neck and in the middle of the Greater Trochanter top. Position the last set of points on the end of the prosthesis and place the corresponding end point in the middle of the femur shaft in the sagittal view.

![Point registration dialog](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/0300033F_364x189.png)

| Points 1                                                     | Points 2                                                     | Points 3                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![Points1](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000340_212x182.png) | ![Points2](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000341_104x182.png) | ![Points3](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000342_105x184.png) |

#### Reposition the implant

The position of the implant can be fine tuned using the reposition tools. In the STL tab right click on the implant and select the move tool ![Move tool](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/02000343.jpg). In the move dialog select Move along inertia axis from the dropdown box. By grabbing one of the arrows you can move the implant in the direction of the selected arrow.

![Reposition move dialog](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000344.png)

| Move step1                                                   | Move step2                                                   | Move step3                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![Move1](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000345_136x248.png) | ![Move2](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000346_133x247.png) | ![Move3](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000347_136x248.png) |

The position of the implant can be verified in both, 2D and 3D views. To visualize the implant in 2D enable the contours by selecting the eye in the contour column of the Objects tab.

![Contours in 2D](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000348.png)

To make the implant visible in the 3D view enable the transparency from the 3D toolbar ![3D transparency](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/02000349_22x22.jpg).

### Ostectomy of the femoral head

To remove the femoral head we will use the polyplane cut from the **3D Tools** menu. Go to **3D Tools > Cut > With Polyplane** ![Polyplane cut](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/0200034C_25x22.jpg) in the menu. In the simulation dialog select the 3D model of the bone, **Yellow**. To perform the cut click once on the top of the femoral neck, turn the 3D and double click on the bottom. This will create a cutting plane as shown in the images below:

![Cutting plane dialog](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/0300034D.png)

| Cut top                                                      | Cut bottom                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![Cut top](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/0300034E_224x193.png) | ![Cut bottom](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/0300034F_224x192.png) |

The orientation of the cut can still be modified. Hover over the center of the red arrow, when the cursor changes into the reposition icon ![Reposition cursor](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000350_25x25.png), hold the left mouse button. By moving the mouse you can change the orientation of the cutting plane.

![Adjust cut plane](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000351_293x251.png)

*Hold the left mouse button to change the orientation of the cutting plane*

To finalize the cut the cutting plane should go completely through the bone. Therefore the depth needs to be increased. In the cut with PolyPlane dialog click on properties. In the properties dialog change the depth to 50 mm.

| Depth increase dialog                                        | Depth increased                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![Depth dialog](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000352.png) | ![Depth result](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000353_221x170.png) |

Click on **OK** to finish the cut.

The cut will create a new 3D model, PolyplanCut-Yellow. To split this model, go to **3D Tools > Split** in the menu. In the **Split** dialog select the PolyplaneCut-yellow 3D model and select largest part. In this way you will only preserve the shaft of the femur.

![Split dialog](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000355_393x153.png)

| Before split                                                 | After split                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![Before split](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000356_224x192.png) | ![After split](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000357_224x192.png) |

### Remesh of the femur and implant

The femur and the implant now have to be remeshed in 3-matic. To do this, go to the **FEA -> Remesh** in the menu. This will bring up the following dialog:

![Remesh both](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000358_448x368.png)

Select both the implant and the shaft of the femur and click on **OK**.

In this step the Parts need to be remeshed to be optimal for FEA purposes. To get the optimal result it is common to follow the next steps in 3-matic:

- Create Non Manifold Assembly
- Define shape parameters
- Inspect the quality of the surface mesh
- Reduce the details of the anatomical part
- Remesh the surface elements
- Generate the volume mesh
- Visualize volume elements
- Analyze Mesh quality

For details on how to get the optimal mesh for your Part see the **Chapter 5: Remesh** of the Tutorials in 3-matic. You can find it under **Help** -> **Tutorial...** in 3-matic.

When you are satisfied with the quality of the mesh copy your object by selecting your object and pressing Ctrl+C on your keyboard. Open your Mimics window and paste your object there by pressing Ctrl+V. The volume mesh will be available in the FEA mesh tab in the project management section. These meshes can then be exported to your FEA software.

### Exporting the Volume mesh

Now you can export the volume mesh from Mimics to a Patran neutral, Abaqus, or ANSYS file. To do this go to the File->Export menu and choose the correct format to export the mesh. Select the FEA meshes and click Add. To export, click OK.

![Export volume mesh](Resources/Images/Mimics 15.0 Reference Guide_track_v2_no_numbers/03000373.png)