# GEOTADAM: Anomaly Detection in High-Dimensional Data

This repository contains the code implementation for GEOTADAM, a geometric-transformation classification-based anomaly detector combined with an analytical margin. The aim of this project is to improve anomaly detection in high-dimensional data, such as images and tabular data, by reducing model complexity and enabling the detection of anomalies in diverse datasets of varying sizes.

## Abstract

Anomaly detection is a vigorously studied area in various industries, including the medical and construction industries. Classification-based techniques are one of the extensively-used for anomaly detection, but they have struggled with detecting anomalies in high-dimensional data. To detect anomalies in high-dimensional data (e.g., image and tabular data), geometric transformation-based and affine transformation-based approaches have been introduced. Despite the feasibility of transformation-based approaches, it is still challenging to identify crucial feature representations (i.e., transformations), thereby increasing model complexity. It is necessary to identify redundant features and reduce model complexity to process much larger-scale data. Therefore, this project aimed to follow a previous work on anomaly detection in high-dimensional data and improve it with the introduction of an analytical margin for effective model reduction. The authors proposed a geometric-transformation classification based anomaly detector combined with an analytic margin (GEOTADAM) by replacing the original margin term with an analytical margin. The receiver operating characteristic-area under the curve (ROC-AUC) and f1 score were used for the performance evaluation according to the data type. The authors compared the results with the previous one and found that GEOTADAM is better in both CIFAR-10 image data and the thyroid tabular data. Moreover, the authors tested GEOTADAM with the few-shot condition (training with 16 images from the STL-10 dataset) and showed better performance than the prior work. The main contribution of this research is to reduce model complexity of transformation-based anomaly detectors and to apply a feature space selection technique to enable end-users to detect anomalies in diverse datasets of varying sizes: superlarge-scale and small-scale.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/GEOTADAM.git
   cd GEOTADAM/
