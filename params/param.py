"""Params for ADDA."""

# params for dataset and data loader
data_root = "data"
process_root = "data/processed"

# params for source dataset
encoder_restore = "snapshots/DANN-encoder-final.pt"
class_classifier_restore = "snapshots/DANN-class-classifier-final.pt"
domain_classifier_restore = "snapshots/DANN-domain-classifier-final.pt"

# params for setting up models
model_root = "snapshots"

# params for training network
num_gpu = 1
manual_seed = None

# params for optimizing models
d_learning_rate = 1e-4
c_learning_rate = 1e-4
beta1 = 0.9
beta2 = 0.999
