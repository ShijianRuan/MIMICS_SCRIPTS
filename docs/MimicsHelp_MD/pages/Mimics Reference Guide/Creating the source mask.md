## Creating the source mask

The smart expand tool needs a source mask from which it grows. The growth is limited to the boundaries or gray value gradients in the images.

To create a source mask, first create an empty mask. This can be done by clicking on the New mask button in Project Management tab.

![](Mimics Reference Guide/../Resources/Images/image28.png)

When you click the New button, the threshold window will open. The threshold will not be used during the segmentation process, but it is used when calculating the 3D model. A good selected threshold will result in a better looking 3D. In this case we will use a minimum threshold of -50 and a maximum threshold of 280.

![](Mimics Reference Guide/../Resources/Images/image29.png)

Next, select the just created mask and click on �Clear Mask� button in the project management, see below. This will clear the mask you have selected. Now you have an empty mask and we will edit this to create the input for the Smart Expand tool.

![](Mimics Reference Guide/../Resources/Images/image30.png)
