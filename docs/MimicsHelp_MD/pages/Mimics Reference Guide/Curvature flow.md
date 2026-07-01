### Curvature flow

The Curvature flow filter performs an edge-preserving smoothing on the images. The iso-contours of the images are viewed as level sets, where the pixels with a particular gray value form one level set. The diffusion speed is proportional to the curvature of the contours. Therefore, areas of high curvature will diffuse faster than areas with low curvature. Hence, small jagged noise artifacts disappear quickly, while large scale artifacts evolve slowly, thereby preserving sharp boundaries between objects.

You can specify two parameters: the number of iterations to be performed and the time step used in the computation of the level set evolution. The typical value for the time step in is 0.125. The number of iterations can usually be around 10. 

![](Mimics Reference Guide/../Resources/Images/CurvatureFlow.png)
