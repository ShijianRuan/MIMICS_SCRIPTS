## Obturator

In the previous cases we have segmented bone structures, whereas in this project we are going to make a soft tissue model. An interesting application is the modeling of the soft tissue around the cavity of the mouth. Such a model can be used as a mold for obturator prostheses. In the case study following this introduction we will do just that.

How are we going to model this soft tissue? Since we are only interested in the area around the cavity, we need to limit the model to the region of interest. By erasing one layer from the active mask in every direction, the cavity and the soft tissue around it will be separated from the rest of the image. This way the region of interest is captured in a 3D box delimited by the removed layers. Next we perform a region growing that starts in the region of interest. Because this region is separated from the active mask, only this area will be put into a new mask after the region growing is done. From the new mask a 3D model can be calculated which will contain just the cavity of the mouth and the soft tissue surrounding it.

The topics that will be discussed in this tutorial are:

  * Case Study


  * Preparation of the data


  * Windowing


  * Orientation


  * Thresholding


  * Editing


  * Region growing


  * View of the end result


