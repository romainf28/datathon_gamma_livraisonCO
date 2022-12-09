"""Init file of the package."""
import logging
from .logging_formatter import CustomFormatter

from .data.window.generator import WindowGenerator
from .models.base.template import Predictor
from .models.wrappers.residual import ResidualWrapper
from .models.autoregressive import FeedBack
from .models.baseline import Baseline, MultiStepLastBaseline, RepeatBaseline
from .data.pipeline import do_datasets
from .benchmark.plot import plot_training
from .benchmark.make_vector import y_vectors
from .pipelines.prediction_pipeline import forecast
from .pipelines.training_pipeline import window_from_data, training_model
#from .pipelines.hparam_pipeline import tensorboard_gen
#from .pipelines.hparam_pipeline_temp import tensorboard_gen_mono
# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)