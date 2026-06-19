# config.py
class SystemConfig:
    CAMERA_INDEX = 1
    SEQUENCE_LENGTH = 10  # Frames for RNN temporal memory
    VIT_MODEL_NAME = 'google/vit-base-patch16-224-in21k'
    RNN_HIDDEN_SIZE = 256
    NUM_COGNITIVE_STATES = 3
    GMM_HISTORY = 500
    WRONSKIAN_THRESHOLD = 0.4
    RESIZE_DIM = (224, 224)