It turns out that you need to specify a random image size in the input model. Then it doesn't matter what the actual image size is if you pass the image to mlContext.Transforms.ResizeImages.

ImageFromRamStrange works good.
