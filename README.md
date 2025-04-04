# CCNN-BP: Continuous, Contactless Blood Pressure Monitoring Using Complex CEEMDAN and Neural Networks

This repository provides a portion of the **dataset** and **neural network code** used in the research paper:

> **"A Novel Contactless Approach to Continuous Blood Pressure Monitoring Using Complex CEEMDAN and Neural Networks"**

---

## 📄 Paper Abstract

*Contactless blood pressure (BP) monitoring is essential for daily healthcare. Existing methods based on radar or remote photoplethysmography (rPPG) typically estimate only systolic blood pressure (SBP) and diastolic blood pressure (DBP). However, continuous monitoring of the complete blood pressure waveform provides deeper insights into cardiovascular health.*

*This paper proposes a novel contactless continuous blood pressure monitoring algorithm termed **CCNN-BP** (Complex CEEMDAN and neural networks for blood pressure monitoring). Radar signals reflected from the chest are decomposed using Complex CEEMDAN (complete ensemble empirical mode decomposition with adaptive noise), extracting heartbeat-related intrinsic mode functions (IMFs). These IMFs are fed into a specialized neural network trained to estimate continuous blood pressure waveforms with accurate morphology. The final blood pressure waveform is obtained by adjusting the amplitude through calibration factors in combination with waveform characteristics.*

*The proposed algorithm was trained and validated using an open-source dataset from Hamburg University of Technology. It achieves a mean absolute error (MAE) of 4.264 mmHg and root mean square error (RMSE) of 6.769 mmHg for amplitude estimation, a MAE of 25.4 ms and RMSE of 33.5 ms for peak delay, and a Pearson correlation coefficient of 0.890. These results demonstrate the feasibility of CCNN-BP for accurate, continuous, and contactless blood pressure waveform monitoring.*

---

## 📁 Contents

This repository includes:

- 🧠 Neural network code used for waveform estimation  
- 📊 Sample data for demonstration and validation purposes  

> 📌 *Note: For full dataset access or detailed implementation, please refer to the corresponding sections in the paper or contact the authors directly.*

---

## 🚀 Getting Started

To reproduce basic experiments:

1. Clone this repository  
2. Ensure you have the required dependencies 
3. Run the neural network training/inference script on the provided sample data

---

## 📚 Citation

If you find this work useful, please consider citing the paper:

