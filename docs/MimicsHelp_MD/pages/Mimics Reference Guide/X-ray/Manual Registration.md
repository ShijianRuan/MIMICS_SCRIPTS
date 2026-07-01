# Manual Registration

![](Mimics Reference Guide/X-ray/../../Resources/Images/Manual%20Registration.png)

The Manual Registration tool allows manual positioning of an X-ray object or Part in the 3D space.

  * Select �X-ray Object Registration� as the registration type from the drop-down menu to move the X-ray object around a selected Part.
  * Select �3D Object Registration� as the registration type to move the selected Part while the X-ray Object remains fixed.


![](Mimics Reference Guide/X-ray/../../Resources/Images/X-ray/MR_DialogBox_Annotations-01_626x350.png)

First, the desired registration type needs to be selected in the drop-down menu. This will define the moving and fixed entities. Checking the initial alignment checkbox will align the Part�s center of mass and the centerline of the X-ray Object. After pressing **OK** , a set of buttons will appear in the bottom right corner of the 3D viewport. These buttons allow translating and rotating the X-ray Object or Part respectively.

Note: Although �X-ray Object Registration� is selected, the Part will appear to move. In fact the X-ray object will move �behind the scenes�. This inverse visualization is implemented because rotating the Part makes it easier to predict the transformation of the contour projection.

_Pivot Point_

The pivot point represents the center around which the X-ray Object or 3D Object rotate. The position of the pivot point can be moved to facilitate the manual registration procedure.

There are 3 options to move the position of the pivot point:

  * Indicate: Position the pivot point by clicking on the desired location.
  * Translate: Move the location of the pivot point by manipulating the translate handles.
  * Reset: The pivot point is reset to the center of mass of the selected 3D Object.


![](Mimics Reference Guide/X-ray/../../Resources/Images/X-ray/MR_Overview_762x285.png)

_Manual Registration Buttons_

The manual registration buttons enable the translation and rotation of the selected moving entity. The functionality of each of the buttons is annotated in the picture below.

![](Mimics Reference Guide/X-ray/../../Resources/Images/X-ray/MR_Controls_Annotated_72dpi-01.png)

Tip: In order to quickly change the step increment value, scroll the mouse wheel while hovering over the step increment text box.

Manual registration can also be performed by using the keyboard. By toggling the CTRL or the NUM5 button, the arrow buttons will switch between the translation and rotation functionality.

![](Mimics Reference Guide/X-ray/../../Resources/Images/X-ray/X-rayMR_Shortcuts_v3.png)

_Align view with X-ray_

The �Align view with X-ray� button at the bottom left of the manual registration buttons indicates whether the 3D viewport should keep the alignment with the X-ray or not. If the button shows this icon ![](Mimics Reference Guide/X-ray/../../Resources/Images/AlignViewportWithXRO_06.png), the viewport will keep the view direction orthogonal to the X-ray. If the button is toggled and this icon ![](Mimics Reference Guide/X-ray/../../Resources/Images/AlignViewportWithXRO_unpressed.png) is shown, it is possible to freely rotate the scene and perform manual registration.

Tip: By default, only the outline of the object is projected on the X-ray. Projecting all contours (including contour outlines of features inside the projected outline) can help performing Manual Registration. This can be toggled with the buttons on the right hand side of the 2D X-ray viewport.

![](Mimics Reference Guide/X-ray/../../Resources/Images/X-ray/MR_Contour-projection-01.png)

|   
---|---
