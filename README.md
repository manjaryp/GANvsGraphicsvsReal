# GAN Graphics Real
<img src= 'https://github.com/manjaryp/GANvsGraphicsvsReal/blob/main/images/index_pic.png' style="max-width: 100%;"> <sup>[image source](#myfootnote1)</sup>

**Distinguishing Natural and Computer-Generated Images using Multi-Colorspace fused EfficientNet** </br>
Manjary P Gangan, Anoop K, and Lajish V L </br>
Computational Intelligence and Data Analytics (CIDA Lab) </br>
Department of Computer Science </br>
University of Calicut, India

:memo: Paper: https://arxiv.org/abs/2110.09428 </br>
:earth_asia: Link: https://dcs.uoc.ac.in/cida/projects/dif/mceffnet.html

**Abstract**: The problem of distinguishing natural images from photo-realistic computer-generated ones either addresses _natural images versus computer graphics_ or _natural images versus GAN images_, at a time. But in a real-world image forensic scenario, it is highly essential to consider all categories of image generation, since in most cases image generation is unknown. We, for the first time, to our best knowledge, approach the problem of distinguishing natural images from photo-realistic computer-generated images as a three-class classification task classifying natural, computer graphics, and GAN images. For the task, we propose a Multi-Colorspace fused EfficientNet model by parallelly fusing three EfficientNet networks that follow transfer learning methodology where each network operates in different colorspaces, RGB, LCH, and HSV, chosen after analyzing the efficacy of various colorspace transformations in this image forensics problem. Our model outperforms the baselines in terms of accuracy, robustness towards post-processing, and generalizability towards other datasets. We conduct psychophysics experiments to understand how accurately humans can distinguish natural, computer graphics, and GAN images where we could observe that humans find difficulty in classifying these images, particularly the computer-generated images, indicating the necessity of computational algorithms for the task. We also analyze the behavior of our model through visual explanations to understand salient regions that contribute to the model's decision making and compare with manual explanations provided by human participants in the form of region markings, where we could observe similarities in both the explanations indicating the powerful nature of our model to take the decisions meaningfully. 

For other inquiries, please contact: </br>
Manjary P Gangan at :email: manjaryp_dcs@uoc.ac.in :earth_asia: [website](https://dcs.uoc.ac.in/~manjary/) </br>
Anoop K at :email: anoopk_dcs@uoc.ac.in :earth_asia: [website](https://dcs.uoc.ac.in/~anoop/)</br>
Lajish V L at :email: lajish@uoc.ac.in :earth_asia: [website](https://dcs.uoc.ac.in/index.php/dr-lajish-v-l)

## Citation
```
@misc{gangan2021distinguishing,
      title={Distinguishing Natural and Computer-Generated Images using Multi-Colorspace fused EfficientNet}, 
      author={Manjary P Gangan and Anoop K and Lajish V L},
      year={2021},
      eprint={2110.09428},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement
This work was supported by the Women Scientist Scheme-A (WOS-A) for Research in Basic/Applied Science from the Department of Science and Technology (DST) of the Government of India 



</br></br>
<a name="myfootnote1">*</a> *`image source`: The first (leftmost) image is GAN generated using StyleGAN2, available at: [link](https://github.com/NVlabs/stylegan2), accessed: 2021-10-21 || the second (middle) image is Computer Graphics generated, available at: [link](https://cgsociety.org/c/featured/1f9s/the-forever), accessed: 2021-10-21 || and the third (rightmost) is a natural image, available at: Computer Graphics versus Photographs dataset [link](https://doi.org/10.1016/j.jvcir.2013.08.009), accessed: 2021-10-21* 
