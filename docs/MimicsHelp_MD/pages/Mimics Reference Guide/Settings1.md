#### Settings  
  
You can choose to do a Global Registration or a Local Registration. In general, you want to do a global registration first and if you would then like to do a feature-based registration, you can use the local registration.

##### Global Registration

The global registration will work on your whole STL. By using the Minimal point distance filter, you can choose not to use all points of your STL. If you e.g. enter a filter value of 1mm, only points that have a distance that is higher than 1mm from each other will be taken into account.

Using a filter value will make sure that your result is independent of the density of the STL and will speed up calculations. If you don�t want to use the filter, you can just give it a value of 0mm.

##### Local Registration

The local registration will only work on all points that are within a certain distance of the border of your mask. By specifying the Maximal distance to mask border filter, you can define this distance.

The local registration will only work if your STL is already positioned correctly.

##### Residual Error

The residual error gives you an indication of how good the position of your STL is with relation to the mask, after doing a registration. The error is a qualitative error, so you can only use this value for comparing the result when changing the parameters of the registration.

Note: Before doing an STL registration, make sure that your STL is already located in the neighborhood of your mask by using the Translate and Rotate button on the STLs tab in the Project Management.
