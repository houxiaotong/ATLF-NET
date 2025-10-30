## ATLF-Net

## 📝Low-rank fused modality assisted magnetic resonance imaging reconstruction via an anatomical variation adaptive transformer



> **Abstract:** In clinical practice, precise magnetic resonance imaging (MRI) reconstruction from undersampled data is crucial. While multi-modal approaches can enhance reconstruction quality, acquiring fully sampled auxiliary information is often time-consuming. Given that computed tomography (CT) images are routinely obtained during clinical examinations, this paper utilizes an anatomical variation adaptive transformer (AVAT) assisted by low-rank fused CT-MRI modality to propose an MRI reconstruction network (ATLF-Net). Specifically, this method leverages a CT-MRI fused modality to assist in MRI reconstruction. The ATLF-Net encompasses fusion and reconstruction processes. The fusion process aims to generate a CT-MRI fused modality that minimizes the gap between it and the MRI modality, serving as auxiliary information for reconstruction. The proposed ATLF-Net comprises the global feature-aware block (GFAB), the local feature-aware block (LFAB), and the low-rank fusion module (LRFM). GFAB and LFAB extract global and local information from shallow features, respectively. LRFM fuses CT and undersampled MRI through modality-specific low-rank factors. During the reconstruction process, an AVAT is developed to extract complex and elongated pathological features. Extensive experiments show that the proposed ATLF-Net achieves robust performance and high-quality reconstructed images with few parameters compared to benchmarks, across various acceleration rates on public datasets.
⭐If this work is helpful for you, please help star this repo. Thanks!🤗


## 📑 Contents

- [Visual Results](#visual_results)
- [News](#news)
- [Installation](#installation)
- [Results](#results)
- [Datasets](#Datasets)
- [Citation](#cite)



## <a name="Real-S"></a> 🥇 The proposed ATLF-Net architecture




## <a name="news"></a> 🆕 News

- **2025-10-29:** ⏰The code is being uploaded. 😄
 


## <a name="installation"></a> Installation

This codebase was tested with the following environment configurations. It may work with other versions.

- CUDA 11.7
- Python 3.9
- PyTorch 1.13.1 + cu117
- NVIDIA 3090 GPU (24 GB) 

All alignments in this study were performed using [ITK-SNAP](http://www.itksnap.org/).  Readers are free to employ alternative registration or annotation tools and pipelines;  our method is software-agnostic, provided that the resulting outputs conform to the required input formats.


One can also create a new anaconda environment, and then install necessary python libraries with this [requirement.txt](https://drive.google.com/file/) and the following command: 
```
conda install requirements.txt
```

## Results
We achieve state-of-the-art performance on various dataset. Detailed results can be found in the paper.

<details>
<summary>Evaluation on AANLIB dataset (click to expand)</summary>
<p align="center">
    <img src="Reconstruction result/reconstruction result-1.png" style="border-radius: 15px">
</p>
</details>

<details>
<summary>Failure cases (click to expand)</summary>
<p align="center">
    <img src="Reconstruction result/Failure cases.png" style="border-radius: 15px">
</p>
</details>

##🗂️ Datasets

[AANLIB](https://www.med.harvard.edu/aanlib/home.html) dataset offered by Harvard Medical School in the United States, is a comprehensive whole brain atlas that is primarily categorized into normal and disease-specific brain images. The AANLIB dataset comprises various imaging modalities, including MRI, CT, PET, and SPECT. Specifically for MR images, the AANLIB dataset provides both T1- and T2-weighted images. 

[CHAOS](https://paperswithcode.com/dataset/CHAOS) dataset, introduced in the ISBI 2019 challenge, is one of the classic benchmarks for abdominal medical image segmentation. It provides paired multi-modal CT and MR data. Specifically, the CHAOS dataset comprises 40 paired CT and MR images, out of which only 20 are annotated and designated as the training set, while the remaining 20 images are unlabeled. Although the CT and MR data are paired, they are not registered. Therefore, preprocessing and registration of the dataset are necessary steps.

[MM-WHS](https://zmiclab.github.io/zxh/0/mmwhs/) dataset, introduced in the MICCAI 2017 challenge, comprises 120 multi-modal cardiac images, including 60 CT/CTA and 60 MRI scans. The data cover the entire heart and its major substructures, acquired under real clinical conditions with varying image quality. Such diversity ensures a realistic assessment of algorithm robustness in clinical applications.

## <a name="cite"></a> 🥰 Citation

Please cite us if our work is useful for your research.

##
## <a name="ack"></a> 🧩 Acknowledgement
 
##
## <a name="con"></a>☎️ Contact

If you have any questions, feel free to approach me. 📞
##


