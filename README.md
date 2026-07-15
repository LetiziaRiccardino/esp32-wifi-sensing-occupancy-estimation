# ESP32 WiFi Sensing for Occupancy Estimation

An end-to-end, device-free framework for indoor occupancy estimation and crowd counting using WiFi Sensing. This repository contains the data processing pipeline to structure raw signal acquisitions and the machine learning classifiers used to estimate the number of people in an environment.

---

## 📌 Project Overview

Traditional crowd counting methods often rely on cameras or dedicated infrastructure (which is highly expensive). This project implements a privacy-preserving, low-cost alternative by leveraging **WiFi Sensing**. 

Using signal disturbances caused by the physical presence and movement of people, this system collects raw wireless data via ESP32 microcontrollers, processes it into structured datasets, and applies supervised Machine Learning algorithms to classify room occupancy levels.

---

## 🗂️ Repository Structure & Core Files

The project is organized around **4 core execution scripts** that cover the entire analytical pipeline:

### 1. Data Processing & Dataset Creation
* **`extract_data.py`** *(or your specific filename)*: This script takes the raw, unstructured data stream extracted from the ESP32 nodes, cleans it, extracts relevant features, and aligns/orders the observations in a structured format to generate the final dataset (e.g., in `.csv` format) ready for training.

### 2. Machine Learning Classifiers
Once the structured dataset is built, three different supervised learning algorithms are implemented and evaluated for the occupancy classification task:
* **`knn_30s.py`**: classifies occupancy levels using the **k-Nearest Neighbors (k-NN)** algorithm, baseline distance-based classification.
* **`svm_30s.py`**: uses **Support Vector Machines (SVM)** with optimized kernels to find optimal decision boundaries for different crowd sizes.
* **`random_forest_30s.py`**: an ensemble method using **Random Forest (RF)** to provide robust predictions and evaluate feature importance.

---

## 🛠️ Pipeline Workflow

```text
[ Raw ESP32 Signal Data ] ──> [ dataset_creator.py ] ──> [ Structured Dataset ]
                                                                   │
                                        ┌──────────────────────────┼──────────────────────────┐
                                        ▼                          ▼                          ▼
                                [ knn_classifier.py ]      [ svm_classifier.py ]     [ random_forest.py ]

```
---

## ⚖️ Acknowledgements & Research Baseline

This repository builds upon the foundational work of **[ESPectre](https://github.com/francescopace/espectre)** by Francesco Pace, an open-source system designed for Wi-Fi-based motion detection using ESP32 microcontrollers. 

### My Contribution & Extension:
* **From Binary Detection to Quantitative Estimation:** While *ESPectre* is designed for binary presence and motion detection, this thesis extends the research towards **occupancy estimation with discrete levels of crowding** (estimating the specific number of people in an environment).
* **Machine Learning Pipeline:** I designed and implemented an analytical pipeline that takes the signal data extracted through the ESPectre-based baseline, structures it into an ordered dataset, and trains supervised Machine Learning models (k-NN, SVM, and Random Forest) to evaluate and compare their classification performance.
