#### Starting from the Export menu

By selecting Iges in the Export menu, a dialog is displayed where you can choose to export the contours of a mask or a 3dd file as an Iges file. The entities used are of type 106 and form 12 (linear path entity).

The advantage of starting the Iges export from a mask is that you can create extra contours (inter-layer) due to interpolation (see the �layer thickness� parameter on the RP Slice calculation parameters page). The disadvantage is that all contours of your mask will be exported. Although, there are filters available to filter out small contours and data points of contours in order to reduce the file size.

Note: If you only want the contours of the mask, make sure the �layer thickness� parameter is set to the slice distance of the image set.

If you want to maintain the original position in space, make sure the First Layer height parameter is set to the first table position. (Otherwise the contours will be translated to this value)
