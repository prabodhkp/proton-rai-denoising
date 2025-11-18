"""
Streamlit Web App for Proton RAI Signal Denoising
Interactive interface for the Proton RAI denoising pipeline

Compatible with:
- Python 3.10.18
- TensorFlow 2.10.0
- Streamlit 1.51.0
"""

import streamlit as st
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from io import BytesIO
import time
import scipy.io as sio

# Import your custom modules
from preprocessing_utils import (
    load_test_data, apply_delay_padding, zero_dead_channels,
    bandpass_filter, downsample_data, crop_or_pad
)
from model_utils import load_model_and_normalization

# Set page config
st.set_page_config(
    page_title="Proton RAI Denoising Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure Streamlit server settings for larger file uploads
# This needs to be set early
import streamlit.config as config
try:
    config.set_option('server.maxUploadSize', 2048)  # 2GB in MB
except:
    pass

# Custom CSS for better styling and compact layout
st.markdown("""
    <style>
    /* Reduce top padding */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    /* Compact headers */
    h1, h2, h3 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    .main-header {
        font-size: 1.5rem;
        color: #1f77b4;
        margin: 0;
        padding: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #2ca02c;
        margin-top: 0.5rem;
    }
    .status-box {
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        padding: 0.5rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    /* Reduce expander spacing */
    .streamlit-expanderHeader {
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)



def initialize_session_state():
    """Initialize session state variables"""
    if 'processing_started' not in st.session_state:
        st.session_state.processing_started = False
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'correlations' not in st.session_state:
        st.session_state.correlations = []
    if 'RF' not in st.session_state:
        st.session_state.RF = None
    if 'RFsum' not in st.session_state:
        st.session_state.RFsum = None


def setup_gpu():
    """Configure GPU settings"""
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            return f"✅ GPU detected: {len(physical_devices)} device(s)", "success"
        else:
            return "ℹ️ No GPU detected, using CPU", "info"
    except Exception as e:
        return f"⚠️ GPU configuration issue: {str(e)}", "warning"


def create_sinogram_plot(input_data, ground_truth, prediction, correlation, 
                        pred_num, start_frame, end_frame):
    """Create sinogram comparison plot"""
    fig = plt.figure(figsize=(18, 8))
    
    # Input sinogram
    ax1 = plt.subplot(2, 3, 1)
    vmax_input = np.max(np.abs(input_data))
    im1 = ax1.imshow(input_data, cmap='seismic', aspect='auto',
                     vmin=-vmax_input, vmax=vmax_input)
    ax1.set_title(f'Input (frames {start_frame}-{end_frame} avg)', 
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('Detectors')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Ground truth
    ax2 = plt.subplot(2, 3, 2)
    vmax_gt = np.max(np.abs(ground_truth))
    im2 = ax2.imshow(ground_truth, cmap='seismic', aspect='auto',
                     vmin=-vmax_gt, vmax=vmax_gt)
    ax2.set_title('Ground Truth (RFsum)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Samples')
    ax2.set_ylabel('Detectors')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Prediction
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(prediction, cmap='seismic', aspect='auto',
                     vmin=-vmax_gt, vmax=vmax_gt)
    ax3.set_title(f'Prediction (Corr: {correlation:.4f})', 
                  fontsize=11, fontweight='bold')
    ax3.set_xlabel('Samples')
    ax3.set_ylabel('Detectors')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Signal comparison at detector 121
    detector_idx = 121
    ax4 = plt.subplot(2, 1, 2)
    ax4.plot(input_data[detector_idx, :], 'g-', label='Input', alpha=0.7, linewidth=2)
    ax4.plot(ground_truth[detector_idx, :], 'b-', label='Ground Truth', alpha=0.7, linewidth=2)
    ax4.plot(prediction[detector_idx, :], 'r--', label='Prediction', alpha=0.7, linewidth=2)
    ax4.set_title(f'Signals at Detector {detector_idx}', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Amplitude')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Prediction {pred_num}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def create_correlation_plot(correlations):
    """Create correlation plot"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(1, len(correlations) + 1)
    ax.plot(x, correlations, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax.axhline(0.9, linestyle='--', color='green', label='0.9 threshold', linewidth=2)
    ax.axhline(0.8, linestyle='--', color='orange', label='0.8 threshold', linewidth=2)
    
    ax.set_title('Correlation: Predictions vs Ground Truth', fontsize=14, fontweight='bold')
    ax.set_xlabel('Prediction Number', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Add statistics box
    stats = (f"Mean: {np.mean(correlations):.4f}\n"
             f"Std: {np.std(correlations):.4f}\n"
             f"Min: {np.min(correlations):.4f}\n"
             f"Max: {np.max(correlations):.4f}")
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, va='top',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'),
            fontsize=10, family='monospace')
    
    plt.tight_layout()
    return fig


def predict_in_chunks_streamlit(RF, RFsum, model, norm_params, target_time_samples, 
                                n_avg, progress_bar, status_text, plot_container):
    """Predict in chunks - plots show at the TOP automatically"""
    num_frames = RF.shape[2]
    num_predictions = num_frames // n_avg
    
    status_text.write(f"📊 **Total frames:** {num_frames}")
    status_text.write(f"🔢 **Number of predictions:** {num_predictions}")
    
    X_mean = norm_params['X_mean']
    X_std = norm_params['X_std']
    y_mean = norm_params['y_mean']
    y_std = norm_params['y_std']
    
    predictions = np.zeros((256, target_time_samples, num_predictions))
    correlations = []
    
    for i in range(num_predictions):
        start = i * n_avg
        end = start + n_avg
        
        # Average frames
        input_avg = np.mean(RF[:, :, start:end], axis=2)
        
        # Prepare input
        X_input = input_avg.reshape(1, 256, target_time_samples, 1)
        X_normalized = (X_input - X_mean) / (X_std + 1e-8)
        
        # Predict
        y_pred_normalized = model.predict(X_normalized, verbose=0)
        y_pred = y_pred_normalized * y_std + y_mean
        prediction = y_pred[0, :, :, 0]
        
        predictions[:, :, i] = prediction
        
        # Calculate correlation
        corr = np.corrcoef(prediction.flatten(), RFsum.flatten())[0, 1]
        correlations.append(corr)
        
        # Update progress
        progress = (i + 1) / num_predictions
        progress_bar.progress(progress)
        
        # Update status
        status_text.write(f"✅ **Prediction {i+1}/{num_predictions}** "
                         f"(frames {start}-{end-1}): Correlation = {corr:.4f}")
        
        # Update plot - it will show at the TOP of Step 3
        plot_container.empty()
        
        with plot_container.container():
            st.success(f"### 🎯 Prediction {i+1}/{num_predictions} | Frames {start}-{end-1} | Corr: **{corr:.4f}**")
            
            # Create and display plot with full width
            fig = create_sinogram_plot(input_avg, RFsum, prediction, corr, 
                                       i+1, start, end-1)
            
            # Use full container width for better visibility
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        
        # Small delay
        time.sleep(0.2)
    
    return predictions, correlations


def run_denoising_pipeline(output_folder, rf_file, n_avg, model_file, norm_params_file):
    """Run the complete denoising pipeline"""
    
    # Signal processing parameters
    fc1 = 0.01e6
    fc2 = 1e6
    Fs_model = 13.33e6
    target_samples = 1024
    
    st.markdown("---")
    st.markdown("### 🔄 Processing Pipeline")
    
    # Create containers: 4, 3, 2, 1 (so order is 4→3→2→1 from top to bottom)
    step4_container = st.container()  # Results at top when done
    step3_container = st.container()  # PLOTS HERE - visible during processing
    step2_container = st.container()
    step1_container = st.container()
    
    try:
        # =====================================
        # STEP 1: Load and Preprocess Data
        # =====================================
        with step1_container:
            st.markdown("#### 📥 Step 1: Loading and Preprocessing Data")
            with st.spinner("Loading RF data..."):
                RF, Fs, delay = load_test_data(rf_file)
                st.success(f"✅ Loaded RF data: Shape {RF.shape}, Fs = {Fs/1e6:.2f} MHz, Delay = {delay*1e6:.2f} µs")
            
            with st.spinner("Preprocessing..."):
                RF = zero_dead_channels(RF)
                RF = bandpass_filter(RF, fc1, fc2, Fs)
                RF = downsample_data(RF, Fs, Fs_model)
                RF = crop_or_pad(RF, target_samples)
                RFsum = RF.mean(axis=2)
                
                st.success(f"✅ Preprocessed RF data: Shape {RF.shape}")
                st.info(f"ℹ️ Delay: {delay} seconds")
        
        # =====================================
        # STEP 2: Load Model
        # =====================================
        with step2_container:
            st.markdown("#### 🤖 Step 2: Loading Denoising Model")
            with st.spinner("Loading model and normalization parameters..."):
                model, norm_params = load_model_and_normalization(model_file, norm_params_file)
                st.success(f"✅ Model loaded successfully")
                st.info(f"ℹ️ Model input shape: {model.input_shape}")
        
        # =====================================
        # STEP 3: Run Predictions
        # =====================================
        with step3_container:
            st.markdown("#### 🔮 Step 3: Running Predictions")
            
            # PLOT GOES HERE FIRST - AT THE TOP!
            st.markdown("### 📊 LIVE PLOT (Watch Here!)")
            st.info("👇 The current prediction plot will appear below and update automatically")
            plot_container = st.empty()
            st.markdown("---")
            
            # Progress and status BELOW the plot
            st.markdown("### 📋 Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run predictions with live updates
            predictions, correlations = predict_in_chunks_streamlit(
                RF, RFsum, model, norm_params, target_samples, n_avg,
                progress_bar, status_text, plot_container
            )
            
            # Apply delay padding
            predictions = apply_delay_padding(predictions, delay, Fs_model)
            predictions = crop_or_pad(predictions, target_samples)
            
            st.success(f"✅ All predictions completed! Final shape: {predictions.shape}")
            
            # Show correlation plot
            st.markdown("#### 📈 Overall Correlation Results")
            corr_fig = create_correlation_plot(correlations)
            st.pyplot(corr_fig)
            plt.close(corr_fig)
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Correlation", f"{np.mean(correlations):.4f}")
            with col2:
                st.metric("Min Correlation", f"{np.min(correlations):.4f}")
            with col3:
                st.metric("Max Correlation", f"{np.max(correlations):.4f}")
            with col4:
                st.metric("Std Deviation", f"{np.std(correlations):.4f}")
        
        # =====================================
        # STEP 4: Save Results
        # =====================================
        with step4_container:
            st.markdown("#### 💾 Step 4: Saving Results")
            
            # Create output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            
            # Generate output filename
            base_name = Path(rf_file).stem
            save_name = f"Denoised_{base_name}_nAvg{n_avg}.mat"
            save_path = os.path.join(output_folder, save_name)
            
            with st.spinner("Saving results..."):
                sio.savemat(
                    save_path,
                    {
                        'results': predictions,
                        'Fs_model': Fs_model,
                        'n_avg': n_avg,
                        'original_file': Path(rf_file).name,
                        'correlations': np.array(correlations),
                        'mean_correlation': np.mean(correlations)
                    },
                    do_compression=True
                )
                
                file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
                st.success(f"✅ Results saved to: `{save_path}`")
                st.info(f"📦 File size: {file_size_mb:.2f} MB")
        
        # Store results in session state
        st.session_state.predictions = predictions
        st.session_state.correlations = correlations
        st.session_state.RF = RF
        st.session_state.RFsum = RFsum
        st.session_state.processing_complete = True
        
        return True, save_path
        
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return False, None


def main():
    """Main Streamlit app"""
    
    # Initialize session state
    initialize_session_state()
    
    # Compact header - minimal spacing
    st.markdown('<h3 style="margin:0; padding:0;">🔬 Proton RAI Signal Denoising Pipeline</h3>', 
                unsafe_allow_html=True)
    
    # Sidebar for inputs
    st.sidebar.header("⚙️ Configuration")
    
    # GPU status
    gpu_msg, gpu_status = setup_gpu()
    if gpu_status == "success":
        st.sidebar.success(gpu_msg)
    elif gpu_status == "warning":
        st.sidebar.warning(gpu_msg)
    else:
        st.sidebar.info(gpu_msg)
    
    st.sidebar.markdown("---")
    
    # Input 1: Output folder - Simple text input (no tkinter for cloud)
    st.sidebar.subheader("1️⃣ Output Folder")
    
    # Initialize session state for folder
    if 'output_folder' not in st.session_state:
        st.session_state.output_folder = "/tmp/denoising_results"
    
    # Simple text input for cloud deployment
    output_folder = st.sidebar.text_input(
        "Output folder path:",
        value=st.session_state.output_folder,
        help="Results will be saved here (default: /tmp/denoising_results)"
    )
    
    st.sidebar.info("💡 **Tip:** In cloud, use /tmp/ for temporary storage")
    
    # Input 2: RF data file
    st.sidebar.subheader("2️⃣ RF Data File (.mat)")
    rf_file = st.sidebar.file_uploader(
        "Upload RF data file:",
        type=['mat'],
        help="Upload your .mat file containing RF data"
    )
    
    # Option to use file path instead
    use_file_path = st.sidebar.checkbox("Or enter file path directly")
    if use_file_path:
        rf_file_path = st.sidebar.text_input(
            "Enter RF data file path:",
            help="Full path to your .mat file"
        )
        if rf_file_path and os.path.exists(rf_file_path):
            rf_file = rf_file_path
            st.sidebar.success(f"✅ File found: {Path(rf_file_path).name}")
        elif rf_file_path:
            st.sidebar.error("❌ File not found!")
    
    # Input 3: Number of frames
    st.sidebar.subheader("3️⃣ Number of Frames (n_avg)")
    n_avg = st.sidebar.number_input(
        "Frames to average:",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
        help="Number of frames to average for denoising"
    )
    
    # Input 4: Model file
    st.sidebar.subheader("4️⃣ Denoising Model (.h5)")
    model_file = st.sidebar.file_uploader(
        "Upload model file:",
        type=['h5', 'hdf5'],
        help="Upload your trained denoising model"
    )
    
    # Option to use file path for model
    use_model_path = st.sidebar.checkbox("Or enter model path directly")
    if use_model_path:
        model_file_path = st.sidebar.text_input(
            "Enter model file path:",
            help="Full path to your .h5 model file"
        )
        if model_file_path and os.path.exists(model_file_path):
            model_file = model_file_path
            st.sidebar.success(f"✅ Model found: {Path(model_file_path).name}")
        elif model_file_path:
            st.sidebar.error("❌ Model file not found!")
    
    # Input 5: Normalization parameters file
    st.sidebar.subheader("5️⃣ Norm Parameters (.npy)")
    norm_params_file = st.sidebar.file_uploader(
        "Upload normalization parameters:",
        type=['npy'],
        help="Upload your normalization parameters file"
    )
    
    # Option to use file path for norm params
    use_norm_path = st.sidebar.checkbox("Or enter norm params path directly")
    if use_norm_path:
        norm_params_path = st.sidebar.text_input(
            "Enter norm params file path:",
            help="Full path to your .npy file"
        )
        if norm_params_path and os.path.exists(norm_params_path):
            norm_params_file = norm_params_path
            st.sidebar.success(f"✅ Params found: {Path(norm_params_path).name}")
        elif norm_params_path:
            st.sidebar.error("❌ Params file not found!")
    
    st.sidebar.markdown("---")
    
    # Validate inputs
    all_inputs_valid = all([
        output_folder,
        rf_file is not None,
        n_avg > 0,
        model_file is not None,
        norm_params_file is not None
    ])
    
    # Compact Configuration Summary - collapsed by default to save space
    with st.expander("📋 Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.caption("**Output:**")
            st.text(Path(output_folder).name if output_folder else "❌")
            st.caption("**RF Data:**")
            st.text(Path(rf_file.name).name if hasattr(rf_file, 'name') else 
                   (Path(rf_file).name if isinstance(rf_file, str) else "❌"))
            st.caption("**n_avg:**")
            st.text(str(n_avg))
        with col2:
            st.caption("**Model:**")
            st.text(Path(model_file.name).name if hasattr(model_file, 'name') else 
                   (Path(model_file).name if isinstance(model_file, str) else "❌"))
            st.caption("**Norm Params:**")
            st.text(Path(norm_params_file.name).name if hasattr(norm_params_file, 'name') else 
                   (Path(norm_params_file).name if isinstance(norm_params_file, str) else "❌"))
    
    # Start processing button - no separator line to save space
    
    if not all_inputs_valid:
        st.warning("⚠️ Please provide all required inputs to proceed.")
    
    start_button = st.button(
        "🚀 Start Denoising Process",
        disabled=not all_inputs_valid,
        type="primary",
        use_container_width=True
    )
    
    # Handle file uploads (save temporarily if uploaded via widget)
    if start_button and all_inputs_valid:
        st.session_state.processing_started = True
        
        # Save uploaded files temporarily if needed
        temp_files = {}
        
        try:
            if hasattr(rf_file, 'read'):
                temp_rf_path = os.path.join(output_folder, "temp_rf_data.mat")
                os.makedirs(output_folder, exist_ok=True)
                with open(temp_rf_path, 'wb') as f:
                    f.write(rf_file.read())
                temp_files['rf'] = temp_rf_path
                rf_file_to_use = temp_rf_path
            else:
                rf_file_to_use = rf_file
            
            if hasattr(model_file, 'read'):
                temp_model_path = os.path.join(output_folder, "temp_model.h5")
                with open(temp_model_path, 'wb') as f:
                    f.write(model_file.read())
                temp_files['model'] = temp_model_path
                model_file_to_use = temp_model_path
            else:
                model_file_to_use = model_file
            
            if hasattr(norm_params_file, 'read'):
                temp_norm_path = os.path.join(output_folder, "temp_norm_params.npy")
                with open(temp_norm_path, 'wb') as f:
                    f.write(norm_params_file.read())
                temp_files['norm'] = temp_norm_path
                norm_params_to_use = temp_norm_path
            else:
                norm_params_to_use = norm_params_file
            
            # Run the pipeline
            success, save_path = run_denoising_pipeline(
                output_folder,
                rf_file_to_use,
                n_avg,
                model_file_to_use,
                norm_params_to_use
            )
            
            if success:
                st.balloons()
                st.success("🎉 Denoising completed successfully!")
                
                # Provide download button
                if os.path.exists(save_path):
                    with open(save_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download Results (.mat)",
                            data=f,
                            file_name=os.path.basename(save_path),
                            mime="application/octet-stream",
                            use_container_width=True
                        )
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())
        
        finally:
            # Clean up temporary files
            for temp_file in temp_files.values():
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except:
                    pass
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>Proton RAI Denoising Pipeline v1.0</strong></p>
    <p>Compatible with Python 3.10 | TensorFlow 2.10 | Streamlit 1.51</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
