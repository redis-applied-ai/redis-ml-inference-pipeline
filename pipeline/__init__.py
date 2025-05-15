"""
Redis ML Inference Pipeline package for real-time fraud detection.
"""

__version__ = "0.3.0"

# Import key modules to make them available at the package level
from pipeline.settings import settings
from pipeline.prepare import prepare_system
from pipeline.worker.inference_worker import InferenceWorker
from pipeline.utils.transaction_generator import TransactionGenerator
