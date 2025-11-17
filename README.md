# 🔬 Proton RAI Signal Denoising

Web application for deep learning-based denoising of Proton Radiography signals.

## 🌐 Live Demo

Access the app at: [Your deployed URL will go here]

## 📋 How to Use

1. **Output Folder**: Click "Browse for Output Folder" to select where results will be saved
2. **RF Data**: Enter the path to your RF data file (.mat format)
3. **Frames to Average**: Set the number of frames (n_avg, e.g., 10)
4. **Model File**: Enter the path to your trained model (.h5 format)
5. **Normalization Parameters**: Enter the path to normalization params (.npy format)
6. **Start Processing**: Click "🚀 Start Denoising Process"
7. **Watch Results**: Plots appear in real-time showing denoising progress
8. **Download**: Get your denoised results when processing completes

## 📦 Requirements

Users must provide their own:
- RF data file (.mat)
- Trained model file (.h5) 
- Normalization parameters file (.npy)

## 🖼️ Features

- ✅ Real-time plot visualization
- ✅ Progress tracking with correlation metrics
- ✅ Compact, user-friendly interface
- ✅ Large file support (up to 2GB)
- ✅ Automatic result saving
- ✅ Download processed data

## 🔧 Technical Details

- **Framework**: Streamlit
- **ML Backend**: TensorFlow 2.10
- **Signal Processing**: SciPy
- **Visualization**: Matplotlib

## 📊 Processing Pipeline

1. Load and preprocess RF data
2. Load denoising model
3. Run predictions with visualization
4. Save denoised results

## 🚀 Local Installation

```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/proton-rai-denoising.git
cd proton-rai-denoising

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app_denoise.py
```

## 📝 License

[Add your license here]

## 👥 Contributors

[Add contributors here]

## 📧 Contact

For questions or issues, please contact: [Your contact info]
