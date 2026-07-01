#### Region growing

We want to make a model of the right femur (left in the image set). Therefore use the following steps:

  * Click the Region Growing button or press Ctrl + R.


  * Set the Source to Green (if this is your base mask) and Target to New Mask. Check the Multiple layer box.


  * Click the left mouse button on one point of the right femur (left on the images). The right femur has now been grown into a new mask (Normally if you have started fresh the femur will be in the yellow mask now).


To calculate your 3D, go to Project Management, Masks tab, select the yellow mask and press the Calculate 3D button. The yellow mask will be automatically selected in the Calculate 3D window, but you need to set the Quality to High and press the Calculate button. 

You can find more details about this in **Simon**.
