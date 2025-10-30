# Glitter Detection App

Glitter Detection is a research initiative developed at the University of Rhode Island AI Lab, focusing on microplastic and glitter particle detection in underwater environments using computer vision and machine learning. The project explores advanced hypersegmentation and YOLO-based classification to identify reflective debris in challenging deep-sea imagery.

## Overview

Detecting glitter and microplastics underwater poses unique challenges due to light scattering, turbidity, and small object size. This project builds an algorithm capable of:
	•	Identifying reflective glitter/microplastic fragments.
	•	Distinguishing particles from background noise, sediment, and bubbles.
	•	Operating under low-light and high-distortion conditions typical of deep-sea footage.

## Objectives
	•	Develop a hypersegmentation algorithm to isolate fine-grained features in underwater imagery.
	•	Train and benchmark YOLO-based segmentation models for microplastic/glitter detection.
	•	Integrate this module into URI AI Lab’s marine environmental monitoring system (alongside OceanDetect).
	•	Support ecological studies and debris quantification efforts by providing automated detection tools.

## Technical Stack
### Component	Tools / Frameworks
Programming Language	Python
Computer Vision	scikit-image, OpenCV
Deep Learning	PyTorch, ultralytics/YOLOv8
Data Processing	NumPy, Pandas, Matplotlib
Compute Resources	URI Unity Cluster (GPU nodes)
Version Control	Git + GitHub

## Pipeline
	1.	Data Ingestion
	2.	Pre-Processing (Frame extraction, color normalization, and contrast enhancement.)
	3.	Hypersegmentation (Custom segmentation using scikit-image to refine object boundaries before detection.)
	4.  Image Output

## Preliminary Results
	•	Precision: ~90% across eight targeted marine classes (fish + debris)
	•	Detection Focus: Reflective particles and microplastics in test footage
	•	Next Steps:
	•	Improve segmentation under low-light conditions
	•	Expand dataset diversity (different depths, water conditions)
	•	Deploy as a Streamlit dashboard or API endpoint for live detection


📜 License

This project is developed under the University of Rhode Island AI Lab and is intended for research and educational use only.
For collaborations or inquiries, please contact Kamron Aggor via GitHub or URI AI Lab channels.

Would you like me to add a “How to Train” section (with dataset setup, training commands, and inference examples) to make it deployable on your GitHub like OceanDetect?
