#### Applying the registration

After placing at least 3 landmark points and choosing the desired fusion method, you can apply the registration.

In case of stitching two datasets, it is possible to check the deviation angle prior to the registration by pressing ![](Mimics Reference Guide/../Resources/Images/AnglePreview.PNG) button. The angle preview shows the deviation from the Z-axes based on landmark points placed on the two datasets. The lesser deviation will result in more accurate stitching of the two datasets.

Mimics will then create a new Mimics project that is as big as Dataset 1. Dataset 2 will be resliced and registered and the part intersecting with Dataset 1 will be registered with Dataset 1.
