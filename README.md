# Sistema inteligente con reentrenamiento continuo para la detección adaptativa de patologías cardiovasculares basado en señales ECG

**ADAPT-ECG** | Residencia Profesional — Ingeniería Biomédica | Instituto Tecnológico de Tijuana | Elias Bejarano Lozada #20213057

Sistema inteligente que combina una Red Neuronal Convolucional 1D con reentrenamiento incremental mediante Replay Buffer para la clasificación adaptativa de arritmias cardiacas en señales ECG, utilizando la base de datos MIT-BIH Arrhythmia Database, procesamiento de señales biomédicas y deep learning (PyTorch), integrado con una interfaz interactiva Streamlit.

---

## Descripción del Proyecto

ADAPT-ECG es un sistema modular de ingeniería biomédica diseñado para:

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
