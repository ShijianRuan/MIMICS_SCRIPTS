## CT Gray Scale

CT images are a pixel map of the linear X-ray attenuation coefficient of tissue. The pixel values are scaled so that the linear X-ray attenuation coefficient of air equals -1024 and that of water equals 0. This scale is called the Hounsfield scale after Godfrey Hounsfield, one of the pioneers in computerized tomography. Using this scale, fat is around -110, muscle is around 40, trabecular bone is in the range of 100 to 300 and cortical bone extends above trabecular bone to about 2000.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0200006D.jpg)   


The pixel values are shown graphically by a set of gray levels that vary linearly from black to white. Mimics displays the CT images using up to 256 gray levels if your display setting is true color (24-bit or 32-bit), 128 gray levels if your display setting is 256 color palette, but as few as 32 gray levels if your display setting is high color (16-bit). The mapping of pixel values into gray levels is specified by a level and a width. A gray scale is centered about its level.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0200006E_450x152.jpg)

For example, a level of 0 specifies that water will be displayed as mid-gray. The extent of the gray scale is specified by its width. The default gray scale used by Mimics allows you to see the full range of tissue from air in the maxillary sinus to the densest of cortical bone, but subtle differences in the soft tissue cannot be visualized. If you narrow the gray scale, you can better visualize subtle differences in the soft tissue or trabecular bone, but at the cost of forcing cortical bone to be in one gray level: white. Narrowing the gray scale can help you locate the mandibular canal if it is not easily seen with the default gray scale.
