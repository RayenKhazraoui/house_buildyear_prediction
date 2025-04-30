# 🏠 Dutch House Construction Year Prediction with Transfer Learning (EfficientNetB0)

This project applies **transfer learning** using an **EfficientNetB0** model to predict the **construction year** of houses in the Netherlands, based on Google Street View images.

---

## 📸 Project Overview

### 🔍 Objective
The goal is to estimate the construction year of Dutch homes using exterior visual features captured from Google Maps imagery.

### 📁 Dataset
- **Source**: The **DAG-register** (Datasets Achtergrondgegevens Gebouwen), an open dataset containing:
  - Coordinates (lat/lon) of buildings in the Netherlands
  - Registered **year of construction** for each building
- **Sampling**:
  - 25,000 buildings were randomly selected
  - Street View images were collected for each building using the **Google Maps API**

### 🧭 Camera Orientation
To ensure that each building is properly visible in the Street View image:
- The camera heading was automatically calculated using the angle between the **available Street View location** and the building’s actual coordinates
- This method ensures that the front facade is generally well-captured

---

## 🤖 Model Architecture

- Base model: **EfficientNetB0** with pretrained **ImageNet** weights
- The **top 4 layers** of the model were unfrozen for fine-tuning
- The model outputs **classification logits** over **13 bins** of construction years (each bin covers a decade, e.g. `1900–1909`, `1910–1919`, ..., `2020–2029`)

---

## 📊 Results

- ✅ **Top-1 Accuracy**: ~40%
- 🔢 **Number of bins**: 13
- ⚖️ Random baseline: ~7.7%
- 📉 Most incorrect predictions were within ±1 bin of the actual construction year

This level of accuracy is decent, considering the subtle differences in architectural features across decades.

## 🖼️ Prediction Examples

### 🏠 Example 1
![Prediction 1](prediction_1.jpg)

### 🏠 Example 2
![Prediction 2](prediction_2.jpg)

### 🏠 Example 3
![Prediction 3](prediction_3.jpg)


---

