### Crop Images

You can crop a project by going to the **Image** menu and then choose the Crop Images button. Following window will appear:

![](Mimics Reference Guide/../Resources/Images/Crop_Project.png)

The outlines of a box will appear on your 2D views and 3D view. When you apply the cropping, a new Mimics project will be created for only that part of the dataset in the box. This way it is possible to create a small project for only the structure you are interested in. Since the cropped project will contain smaller slices, the speed of Mimics will be increased while working on the project.

The box can be adjusted by dragging the borders of the box in the 2D views or by adjusting the box properties in the crop interface.

In the following example, we create a new Mimics project for only the femur, starting from a dataset that contains full knee.

![](Mimics Reference Guide/../Resources/Images/CropProject_BoundingBox_846x400.png)

When cropping an image set, Mimics will write out the transformation matrix that was applied during the cropping. The file with the transformation matrix will be saved in the folder of the original project.
