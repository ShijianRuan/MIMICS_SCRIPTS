# Contour-based Registration  
  
![](Mimics Reference Guide/X-ray/../../Resources/Images/Contour-based%20Registration.png)

Contour-based registration will optimize the contour fit between the projected contour of the selected Part and the selected Contour of an X-ray object. Depending on the selected type of registration in the drop-down menu, X-ray Object Registration or 3D Object Registration, the X-ray object or Part will move. A contour-based measurement will automatically be created to provide an indication of the registration accuracy. The contour-based measurement will appear in the PM tab as well as in the log.

! Important: Before launching Contour-based Registration, it is necessary to perform manual registration so the algorithm is able to optimize the registration. The Contour-based Registration feature is not able to converge the distance between the projected and created contour if the selected entities are initially too distant from each other. The contour projection of the selected Part should roughly fit the created contour on the X-ray object. The projected outline should share the same distinctive features as the created contour.

The algorithm will not start if the selected Part is not visible in the 3D viewport or if the Part is not positioned inside the X-ray object pyramid.

![](Mimics Reference Guide/X-ray/../../Resources/Images/ContourBasedRegistration_DialogBox.png)

**X-ray Object Registration**

X-ray Object Registration only allows the selection of one Contour and one Part. The contour can contain multiple segments and does not need to be closed (more info in the Create Contour help chapter). The contour-based algorithm will optimize the fit of the contour segments with the corresponding areas of the contour projection.

![](Mimics Reference Guide/X-ray/../../Resources/Images/CBR_ContourInput_482x370.png) |  ![](Mimics Reference Guide/X-ray/../../Resources/Images/CBR_ContourInput_03_522x369.png)  
---|---  
  
**3D Object Registration**

3D Object Registration will optimize the fit of the selected Contour and contour projection by moving the selected Part. Multiple Contours can be selected as an input to guide the Part registration. The selected Part will be moved so the contour projection will fit best to all the selected contours.
