# PROYECT_ADAPT-ECG

Adaptive ECG intelligence system with continuous retraining for cardiovascular pathology detection using the MIT-BIH Arrhythmia Database, signal processing techniques, and deep learning (PyTorch), integrated with an interactive Streamlit interface.

---

## Project Overview

PROYECT_ADAPT-ECG is a modular biomedical engineering system designed to:

- Load ECG signals from the MIT-BIH Arrhythmia Database (PhysioNet)
- Apply signal preprocessing (bandpass filtering + normalization)
- Perform deep learning inference for arrhythmia detection
- Support adaptive continuous retraining
- Provide interactive visualization through Streamlit

The architecture is designed for research-oriented and scalable AI-based cardiac analysis.

---

## Dataset

This project uses the:

**MIT-BIH Arrhythmia Database**  
PhysioNet: https://physionet.org/content/mitdb/

### Dataset Characteristics

- 48 half-hour two-channel ambulatory ECG recordings
- Sampled at 360 Hz
- Annotated beat-by-beat by cardiologists
- Contains multiple arrhythmia types
- Widely used as benchmark in ECG classification research

### Citation

Moody, G. B., & Mark, R. G. (2001).  
The impact of the MIT-BIH Arrhythmia Database.  
IEEE Engineering in Medicine and Biology Magazine.

Users must comply with PhysioNet's data usage policies.

---

## Repository Structure
