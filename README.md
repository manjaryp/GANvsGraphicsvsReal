# GAN Graphics Real
<img src= 'https://github.com/manjaryp/GANvsGraphicsvsReal/blob/main/images/index_pic.png' style="max-width: 100%;"> <sup>[image source](#myfootnote1)</sup>


## Distinguishing Natural and Computer-Generated Images using Multi-Colorspace fused EfficientNet </br>
Manjary P. Gangan, Anoop Kadan, and Lajish V. L.</br>

Department of Computer Science </br>
University of Calicut, India

:memo: Paper : https://www.sciencedirect.com/science/article/abs/pii/S2214212622001247 </br>
:earth_asia: Link: https://dcs.uoc.ac.in/cida/projects/dif/mceffnet.html

**Abstract**: The problem of distinguishing natural images from photo-realistic computer-generated ones either addresses _natural images versus computer graphics_ or _natural images versus GAN images_, at a time. But in a real-world image forensic scenario, it is highly essential to consider all categories of image generation, since in most cases image generation is unknown. We, for the first time, to our best knowledge, approach the problem of distinguishing natural images from photo-realistic computer-generated images as a three-class classification task classifying natural, computer graphics, and GAN images. For the task, we propose a Multi-Colorspace fused EfficientNet model by parallelly fusing three EfficientNet networks that follow transfer learning methodology where each network operates in different colorspaces, RGB, LCH, and HSV, chosen after analyzing the efficacy of various colorspace transformations in this image forensics problem. Our model outperforms the baselines in terms of accuracy, robustness towards post-processing, and generalizability towards other datasets. We conduct psychophysics experiments to understand how accurately humans can distinguish natural, computer graphics, and GAN images where we could observe that humans find difficulty in classifying these images, particularly the computer-generated images, indicating the necessity of computational algorithms for the task. We also analyze the behavior of our model through visual explanations to understand salient regions that contribute to the model's decision making and compare with manual explanations provided by human participants in the form of region markings, where we could observe similarities in both the explanations indicating the powerful nature of our model to take the decisions meaningfully. 

***For inquiries, please contact:*** </br>
&nbsp; Manjary P. Gangan, University of Calicut, Kerala, India. :email: manjaryp@gmail.com :earth_asia: [website](https://dcs.uoc.ac.in/~manjary/) </br>
&nbsp; Anoop Kadan, University of Southampton, UK. :email: anoopkadan23@gmail.com :earth_asia: [website](https://www.southampton.ac.uk/people/65qvt5/doctor-anoop-kadan)</br>

## Citation
```
@article{gangan2022distinguishing,
  title={Distinguishing natural and computer generated images using Multi-Colorspace fused EfficientNet},
  author={Manjary {P. Gangan} and Anoop K. and Lajish {V. L.}},
  journal={Journal of Information Security and Applications},
  volume={68},
  pages={103261},
  year={2022},
  publisher={Elsevier},
  issn = {2214-2126},
  doi = {https://doi.org/10.1016/j.jisa.2022.103261},
  url = {https://www.sciencedirect.com/science/article/pii/S2214212622001247}
}

```

## Acknowledgement
This work was supported by the Women Scientist Scheme-A (WOS-A) for Research in Basic/Applied Science from the Department of Science and Technology (DST) of the Government of India 



</br></br>
<a name="myfootnote1">*</a> *`image source`: The first (leftmost) image is GAN generated using StyleGAN2, available at: [link](https://github.com/NVlabs/stylegan2), accessed: 2021-10-21 || the second (middle) image is Computer Graphics generated, available at: [link](https://cgsociety.org/c/featured/1f9s/the-forever), accessed: 2021-10-21 || and the third (rightmost) is a natural image, available at: Computer Graphics versus Photographs dataset [link](https://doi.org/10.1016/j.jvcir.2013.08.009), accessed: 2021-10-21* 
