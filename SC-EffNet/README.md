# SC-EffNet


The implementation of Single Colorspace EfficientNet (_SC-EffNet_) using YDbDr colorspace and after rescaling is given in [sceffnet.py](https://github.com/manjaryp/GANvsGraphicsvsReal/blob/main/SC-EffNet/sceffnet.py). The model settings are detailed in `Section III.B` of the paper. For implementing other colorspace transformations the  code given below can be modified with the intended colorspace transformation by refering [opencv colorspace conversions](https://docs.opencv.org/master/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0a353a4b8db9040165db4dacb5bcefb6ea) or [scikit-image colorspace conversions](https://scikit-image.org/docs/dev/api/skimage.color.html). Also, for without rescaling implementations, the function for rescaling (_scale0to255_) can be neglected. 

1. For implementing SC-EffNet<sub>ABC</sub> in ABC colorspace, do the folowing changes.
</br></br>
Using skimage
```python
def colorFunction(image):
    color_transf_image = skimage.color.rgb2<ABC>(image) 
    scaled_image = scale0to255(color_transf_image) 
    return scaled_image
```

</br>Using cv2
```python
def colorFunction(image):
    color_transf_image = cv2.cvtColor(image,cv2.COLOR_RGB2<ABC>)  
    scaled_image = scale0to255(color_transf_image) 
    return scaled_image
```

</br></br>
2. For without rescaling

```python
def colorFunction(image):
    color_transf_image = skimage.color.rgb2<ABC>(image) 
    #scaled_image = scale0to255(color_transf_image) 
    return color_transf_image
```
