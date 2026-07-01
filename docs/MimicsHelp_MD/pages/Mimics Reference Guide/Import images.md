#### Import images

Select New Project Wizard from the File menu. In the New Project Wizard select all the images at "MedData\DemoFiles\RAW_Images" directory and click Next.

![](Mimics Reference Guide/../Resources/Images/Raw_Import_554x438.png)

If the images are in Raw format, the New Project Wizard will automatically take you to the following steps. You can also force raw import by selecting the Raw import option from the dropdown list at the bottom right of the dialog. If this option is selected, Mimics will import all images as RAW images. When you press the Next button, you will see that the files are recognized as �unknown files� in the �Import log� window. Click Next to go to the Raw image properties window.

![](Mimics Reference Guide/../Resources/Images/raw_import_tutorial_491x356.png)

In the Raw image properties window you will have to enter the parameters of the scan, namely, the Scan resolution, the Image parameters and the Pixel properties. This information is usually provided along with the scan by the radiologist. For this case, the Scan resolution is 0.5 X 0.5 X 1 mm and the Image parameters are 128 X 128 pixels. The pixel values are in Signed Long format with Low byte order Byte swapping. When you have entered the correct parameters, you can preview the images and Next button will be activated. 

Following is some more explanation on the parameters. 

##### Scan resolution

Here the sizes of the pixels have to be entered. For this example, each pixel are 0.5mm in X-direction, 0.5mm in Y-direction and 1mm in Z-direction. If the image slices are taken axially, then Z-direction would be equivalent to slice distance. 

##### Image parameters

The file header size is calculated automatically, based on the file size, the resolution of the images and the pixel type.

Typically a file contains both a file header and the image itself (in some rare cases also a footer is present). The file header can contain information about pixel size, patient data, � The image is a matrix of pixels. The horizontal (or vertical) image size is equal to the number of pixels in that direction.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002D7_317x313.png)

The number of pixels in vertical and horizontal section is the height and the width of the images. Common sizes of images are: 256 * 256, 512 * 512 and 1024 * 1024. In this example, the images have a resolution of 256*256.

##### Pixel properties

The number of bytes per pixel depends on the type of the pixel. Some examples of pixel types and their respective sizes (note that these types can be either signed or unsigned, however, this does not affect their size):

  * Byte: 1 byte


  * Short: 2 bytes


  * Long: 4 bytes


  * Float: 4 bytes


If you fill these values in, you will see that Mimics will set the file header size to 8432 bytes.

Byte swapping determines the order in which the images are read. You can try different options for byte swapping parameter and preview the images. For this case, when High byte first is chose, there are local distortions all over the image, because the data is read in the wrong order. 

For this example, the pixel type is Signed Short and Low Byte First for the parameter Byte Swapping. 

##### Study information

Here you may fill in an appropriate name for the patient name. This will be the name that is used for your project.
