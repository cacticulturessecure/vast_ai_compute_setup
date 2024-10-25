import whisperx
import torch

# Device config
device = "cuda" if torch.cuda.is_available() else "cpu"
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device:", device)
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name())

# Disable TF32 for consistency
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Load model (this will verify whisperx is working)
try:
    model = whisperx.load_model("large-v2", device, compute_type="float16" if device == "cuda" else "int8")
    print("\nWhisperX model loaded successfully!")
except Exception as e:
    print("\nError loading WhisperX model:", str(e))
