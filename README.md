# CorrosionAI

CorrosionAI is a deep-learning tool for identifying corrosion-related surface morphologies of amphibole grains in optical photomicrographs. It classifies each grain image into one of five morphology classes and can summarize image-level predictions into a refined corrosion index (CI*) for sample-level comparison.

This repository provides the inference workflow, prediction scripts, Colab notebook, and documentation associated with the CorrosionAI model.


## Using the tool

Using the tool is very simple. Users only need to prepare their grain images,  
click this button to open the inference software in Google Colab:[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tian-haoyan/CorrosionAI/blob/main/colab/grain_classifier_demo.ipynb), 
and follow the step-by-step instructions inside. 
No additional programming or local software installation is required.


The Colab notebook allows users to upload their own images and run the trained model directly in the browser. The output is a single CSV file that reports the predicted corrosion class and class probabilities for each grain image, with a final summary row giving the predicted corrosion composition and the overall refined Corrosion Index (CI*) for the sample.


## Preparing input images

Users are encouraged to prepare grain images following the image-acquisition and preprocessing procedure described in the Methods section of *Automated determination of mineral corrosion features in sand and sandstone by deep learning*. Before running the model, please also check that the images meet the following input requirements.

Recommended images:

- contain one clearly visible amphibole grain，avoid images with multiple overlapping grains, severe blur, strong reflections, uncertain mineral identification;
- have limited background and limited overlap with other grains;
- are cropped as square, single-grain images, preferably 576 × 576 pixels, to match the image format used during model development;
- are RGB optical photomicrographs;
- show a reasonably sharp grain surface;
- do not contain labels, arrows, scale bars, or text covering the grain.



## Classes

The model predicts five classes:

1. `Corroded`
2. `Etched`
3. `Skeletal`
4. `Unweathered-A` — angular unweathered grains
5. `Unweathered-R` — rounded unweathered grains

The class order is fixed and should not be changed when using the checkpoint.

## Refined corrosion index

For a sample containing `N` classified grains, CI* is calculated as:

```text
CI* = 100 x (Skeletal + 0.75 x Etched + 0.5 x Corroded) / N
```

CI* is an image-derived summary index. It should be interpreted together with petrographic, sedimentological, and geological context rather than as a standalone environmental diagnosis.



