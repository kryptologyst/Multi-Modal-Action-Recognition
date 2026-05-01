"""
Interactive demo application for multi-modal action recognition.

This module provides both Streamlit and Gradio interfaces for testing
the multi-modal action recognition system.
"""

import os
import tempfile
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import librosa
from PIL import Image
import streamlit as st
import gradio as gr

from ..config import Config, DemoConfig, DEFAULT_ACTION_CLASSES, SAFETY_DISCLAIMER
from ..models import MultiModalActionRecognitionModel
from ..data import MultiModalActionDataset


class ActionRecognitionDemo:
    """
    Demo application for multi-modal action recognition.
    """
    
    def __init__(self, config: DemoConfig):
        self.config = config
        self.device = self._get_device()
        self.model = None
        self.action_classes = DEFAULT_ACTION_CLASSES
        
        # Load model if path exists
        if os.path.exists(config.model_path):
            self.load_model(config.model_path)
    
    def _get_device(self) -> torch.device:
        """Get the appropriate device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.config.device)
    
    def load_model(self, model_path: str):
        """Load the trained model."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model from config
            model_config = checkpoint['config'].model
            self.model = MultiModalActionRecognitionModel(model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded successfully from {model_path}")
            print(f"Device: {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def preprocess_video(self, video_file) -> torch.Tensor:
        """Preprocess video file for inference."""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                tmp_file.write(video_file.read())
                tmp_path = tmp_file.name
            
            # Load video with OpenCV
            cap = cv2.VideoCapture(tmp_path)
            frames = []
            
            # Read frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            os.unlink(tmp_path)  # Clean up temp file
            
            if not frames:
                raise ValueError("No frames found in video")
            
            # Sample frames uniformly
            num_frames = 16  # Default number of frames
            if len(frames) >= num_frames:
                indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
            else:
                indices = np.arange(len(frames))
                # Pad with last frame
                indices = np.pad(indices, (0, num_frames - len(frames)), mode='edge')
            
            # Select frames and resize
            selected_frames = [frames[i] for i in indices]
            resized_frames = [cv2.resize(frame, (224, 224)) for frame in selected_frames]
            
            # Convert to tensor
            video_tensor = torch.from_numpy(np.array(resized_frames)).float()
            video_tensor = video_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
            video_tensor = video_tensor / 255.0  # Normalize
            
            # Normalize with ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
            video_tensor = (video_tensor - mean) / std
            
            return video_tensor.unsqueeze(0)  # Add batch dimension
            
        except Exception as e:
            print(f"Error preprocessing video: {e}")
            return torch.zeros(1, 3, 16, 224, 224)
    
    def preprocess_audio(self, audio_file) -> torch.Tensor:
        """Preprocess audio file for inference."""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_file.read())
                tmp_path = tmp_file.name
            
            # Load audio
            waveform, sr = librosa.load(tmp_path, sr=16000)
            
            # Pad or truncate to fixed length
            max_length = 160000  # 10 seconds at 16kHz
            if len(waveform) > max_length:
                waveform = waveform[:max_length]
            else:
                waveform = np.pad(waveform, (0, max_length - len(waveform)))
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(waveform).float()
            
            # Normalize
            audio_tensor = audio_tensor / (torch.max(torch.abs(audio_tensor)) + 1e-8)
            
            # Convert to mel spectrogram
            try:
                import torchaudio.transforms as audio_transforms
                mel_transform = torch.nn.Sequential(
                    audio_transforms.MelSpectrogram(
                        sample_rate=16000,
                        n_mels=80,
                        n_fft=1024,
                        hop_length=512
                    )
                )
            except ImportError:
                # Fallback: create dummy mel spectrogram
                mel_transform = torch.nn.Sequential(
                    torch.nn.Linear(len(audio_tensor), 80 * 313)
                )
            
            mel_spec = mel_transform(audio_tensor)
            
            os.unlink(tmp_path)  # Clean up temp file
            
            return mel_spec.unsqueeze(0)  # Add batch dimension
            
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            return torch.zeros(1, 80, 313)  # Default mel spectrogram shape
    
    def predict(self, video_file, audio_file) -> Dict[str, Any]:
        """Make prediction on video and audio."""
        if self.model is None:
            return {
                'error': 'Model not loaded. Please ensure the model checkpoint exists.',
                'predictions': [],
                'confidences': []
            }
        
        try:
            # Preprocess inputs
            video_tensor = self.preprocess_video(video_file)
            audio_tensor = self.preprocess_audio(audio_file)
            
            # Move to device
            video_tensor = video_tensor.to(self.device)
            audio_tensor = audio_tensor.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(video_tensor, audio_tensor)
                logits = outputs['logits']
                confidences = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
            
            # Get top predictions
            top_confidences, top_indices = torch.topk(confidences, k=5, dim=1)
            
            results = {
                'predictions': [self.action_classes[i] for i in top_indices[0].cpu().numpy()],
                'confidences': top_confidences[0].cpu().numpy().tolist(),
                'top_prediction': self.action_classes[predictions[0].item()],
                'top_confidence': confidences[0][predictions[0]].item()
            }
            
            return results
            
        except Exception as e:
            return {
                'error': f'Prediction error: {str(e)}',
                'predictions': [],
                'confidences': []
            }


def create_streamlit_app(config: DemoConfig):
    """Create Streamlit demo application."""
    
    # Page config
    st.set_page_config(
        page_title="Multi-Modal Action Recognition",
        page_icon="🎬",
        layout="wide"
    )
    
    # Title and description
    st.title("🎬 Multi-Modal Action Recognition")
    st.markdown("""
    This demo recognizes human actions from video and audio inputs using multi-modal deep learning.
    Upload a video file and corresponding audio file to get action predictions.
    """)
    
    # Safety disclaimer
    with st.expander("⚠️ Safety Disclaimer", expanded=False):
        st.markdown(SAFETY_DISCLAIMER)
    
    # Initialize demo
    if 'demo' not in st.session_state:
        st.session_state.demo = ActionRecognitionDemo(config)
    
    demo = st.session_state.demo
    
    # Sidebar for model info
    with st.sidebar:
        st.header("Model Information")
        if demo.model is not None:
            st.success("✅ Model loaded successfully")
            st.info(f"Device: {demo.device}")
        else:
            st.error("❌ Model not loaded")
            st.info("Please ensure the model checkpoint exists at the specified path.")
        
        st.header("Supported Formats")
        st.markdown("""
        **Video:** MP4, AVI, MOV  
        **Audio:** WAV, MP3, FLAC  
        **Max Duration:** 30 seconds  
        **Max File Size:** 100 MB
        """)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📹 Video Input")
        video_file = st.file_uploader(
            "Upload video file",
            type=['mp4', 'avi', 'mov'],
            help="Upload a video file containing human actions"
        )
        
        if video_file is not None:
            st.video(video_file)
    
    with col2:
        st.header("🎵 Audio Input")
        audio_file = st.file_uploader(
            "Upload audio file",
            type=['wav', 'mp3', 'flac'],
            help="Upload the corresponding audio file"
        )
        
        if audio_file is not None:
            st.audio(audio_file)
    
    # Prediction button
    if st.button("🔍 Predict Action", type="primary"):
        if video_file is not None and audio_file is not None:
            with st.spinner("Processing video and audio..."):
                results = demo.predict(video_file, audio_file)
            
            if 'error' in results:
                st.error(results['error'])
            else:
                # Display results
                st.header("🎯 Prediction Results")
                
                # Top prediction
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Predicted Action",
                        results['top_prediction'],
                        f"{results['top_confidence']:.2%}"
                    )
                
                with col2:
                    st.metric(
                        "Confidence",
                        f"{results['top_confidence']:.2%}"
                    )
                
                # Top 5 predictions
                st.subheader("Top 5 Predictions")
                for i, (pred, conf) in enumerate(zip(results['predictions'], results['confidences'])):
                    st.progress(conf, text=f"{i+1}. {pred}: {conf:.2%}")
        
        else:
            st.warning("Please upload both video and audio files.")
    
    # Additional information
    with st.expander("ℹ️ About This Demo"):
        st.markdown("""
        This multi-modal action recognition system combines visual and audio information
        to classify human actions. The model uses:
        
        - **Video Encoder:** TimeSformer or CNN-based temporal encoder
        - **Audio Encoder:** Wav2Vec2 or Mel-spectrogram CNN
        - **Fusion Strategy:** Cross-attention between modalities
        - **Action Classes:** Walking, Running, Jumping, Sitting, Standing, etc.
        
        The system is designed for research and educational purposes.
        """)


def create_gradio_app(config: DemoConfig):
    """Create Gradio demo application."""
    
    # Initialize demo
    demo_app = ActionRecognitionDemo(config)
    
    def predict_action(video_file, audio_file):
        """Gradio prediction function."""
        if video_file is None or audio_file is None:
            return "Please upload both video and audio files."
        
        results = demo_app.predict(video_file, audio_file)
        
        if 'error' in results:
            return results['error']
        
        # Format results
        output = f"**Predicted Action:** {results['top_prediction']}\n"
        output += f"**Confidence:** {results['top_confidence']:.2%}\n\n"
        output += "**Top 5 Predictions:**\n"
        
        for i, (pred, conf) in enumerate(zip(results['predictions'], results['confidences'])):
            output += f"{i+1}. {pred}: {conf:.2%}\n"
        
        return output
    
    # Create Gradio interface
    with gr.Blocks(title="Multi-Modal Action Recognition") as interface:
        gr.Markdown("# 🎬 Multi-Modal Action Recognition")
        gr.Markdown("""
        Upload a video file and corresponding audio file to recognize human actions.
        The system combines visual and audio information for accurate action classification.
        """)
        
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Video Input")
                audio_input = gr.Audio(label="Audio Input")
                
                predict_btn = gr.Button("🔍 Predict Action", variant="primary")
            
            with gr.Column():
                output = gr.Markdown(label="Prediction Results")
        
        # Safety disclaimer
        with gr.Accordion("⚠️ Safety Disclaimer", open=False):
            gr.Markdown(SAFETY_DISCLAIMER)
        
        # About section
        with gr.Accordion("ℹ️ About This Demo", open=False):
            gr.Markdown("""
            This multi-modal action recognition system combines visual and audio information
            to classify human actions. The model uses:
            
            - **Video Encoder:** TimeSformer or CNN-based temporal encoder
            - **Audio Encoder:** Wav2Vec2 or Mel-spectrogram CNN  
            - **Fusion Strategy:** Cross-attention between modalities
            - **Action Classes:** Walking, Running, Jumping, Sitting, Standing, etc.
            
            The system is designed for research and educational purposes.
            """)
        
        # Connect the prediction function
        predict_btn.click(
            fn=predict_action,
            inputs=[video_input, audio_input],
            outputs=output
        )
    
    return interface


def run_demo(config: DemoConfig):
    """Run the demo application."""
    if config.demo_type == "streamlit":
        # Streamlit app is run via command line
        print("To run the Streamlit demo, use:")
        print("streamlit run demo/streamlit_app.py")
    elif config.demo_type == "gradio":
        interface = create_gradio_app(config)
        interface.launch(
            server_name=config.host,
            server_port=config.port,
            share=False
        )
    else:
        raise ValueError(f"Unsupported demo type: {config.demo_type}")


if __name__ == "__main__":
    from ..config import DemoConfig
    
    config = DemoConfig()
    run_demo(config)
