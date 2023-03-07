# [MIAI] Facial expression recognition:smile:

# Introduction
Facial expression recognition takes an video frame and classifies into eight categories (neutral, happy, angry, sad, fear, surprise, disgust, contempt).

<img src="https://github.com/keti-iiprc/miai_facial_expression_recognition/blob/master/media/ezgif.com-video-to-gifA.gif?raw=true">
<img src="https://github.com/keti-iiprc/miai_facial_expression_recognition/blob/master/media/ezgif.com-video-to-gifB.gif?raw=true">

# How to use

## Environment Setup
The code is tested in the following environment. The newer version of the packages should be also be fine.
```
python==3.7+
numpy
mediapipe
pandas
opencv-python
```


# Data preparation

Download pretrained model from the link: [google drive](https://drive.google.com/drive/folders/1hcouPIyxpP2MRCy8CdGJWQzIAbzbKDyQ?usp=sharing, "google drive"). 

```
<root>
   - checkpoints
     > ckpt.pth  
     > resnet18_msceleb.pth
   ...
   demo.py
   ...
```

# Inference
```
python demo.py
```



> This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.0000-0-00000, Artificial intelligence research about multi-modal interactions for empathetic conversations with humans)

## 
