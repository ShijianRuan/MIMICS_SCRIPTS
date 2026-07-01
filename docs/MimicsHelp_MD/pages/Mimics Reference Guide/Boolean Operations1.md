#### Boolean Operations

Let�s examine what we�ve obtained so far: the final mandible (with prosthesis), the left prosthesis and the right prosthesis in three different masks. That�s nice, but we said earlier that we would like to have the mandible without prosthesis. We can achieve this with some Boolean operations. Press the Boolean operations button![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/020002AE_25x25.jpg). Let's use the following calculation: 

Mandible without prosthesis = total mandible � left prosthesis � right prosthesis

Follow the steps below:

  * Mask A: final mandible


Operation: minus

Mask B: left prosthesis

Result: new mask (called mask C for reference)

After these options are set, press the Apply button.

  * Mask A: mask C (obtained in the first step)


Operation: minus

Mask B: right prosthesis

Result: new mask

After these options are set, press the Apply button.

Press the Close button. The last mask (the one that should be active now) contains the pixels of the mandible without the prosthesis. Calculate and view the 3D of this mask. Show also the left and right prosthesis. The other 3Ds can be set invisible. You should have a model that looks like this one.
