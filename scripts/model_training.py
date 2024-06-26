import os
import random
import time
import datajoint as dj

fetch_download_path = '/data/fetched_from_attach'
os.environ['TORCH_HOME'] = '/data'
os.environ['DJ_SUPPORT_FILEPATH_MANAGEMENT'] = 'TRUE'

dj.config["enable_python_native_blobs"] = True
dj.config["database.host"] = 'at-database3.ad.bcm.edu'
dj.config['database.user'] = 'kelli'
dj.config['database.password'] = 'database-questions-left'
dj.config['nnfabrik.schema_name'] = "nnfabrik_toy_V4"

schema = dj.schema("nnfabrik_toy_V4")

from nnfabrik.main import *
import nnfabrik
from nnfabrik import main, builder

from mei import mixins 
from mei import legacy
from mei.legacy.ops import ChangeNorm

import nnvision
from nnvision.tables.main import Recording
from nnvision.tables.from_nnfabrik import DataInfo, TrainedModel, TrainedHyperModel, TrainedTransferModel, SharedReadoutTrainedModel
from nnvision.tables.from_mei import MEI, MEIShared, MEISeed, Method, Ensemble, SharedReadoutTrainedEnsembleModel, MethodGroup, MEIObjective, MEITargetUnits, MEITargetFunctions, MEITextures, MEIPrototype
from nnvision.tables.from_mei import TransferEnsembleModel, MEITransfer
from nnvision.tables.ensemble_scores import TestCorrelationEnsembleScore, CorrelationToAverageEnsembleScore
#from nnvision.tables.measures import ExplainableVar
from nnvision.tables.scores import TestCorrelationScore
from nnvision.datasets.utility import get_validation_split, ImageCache, get_cached_loader, get_fraction_of_training_images, get_crop_from_stimulus_location

dj.config['stores'] = {
  'minio': {
    'protocol': 'file',
    'location': '/mnt/dj-stor01/',
    'stage': '/mnt/dj-stor01/'
  }
}

print(dj.config) 

#### add the dataset hash here ######
dataset_hash="072e4345d9a54ad5f487f6d632ba695b"
#################################


# Robust ResNet 3.0 gauss
model_hash_resnet_gauss = 'ade1c26ff74aef5479499079a219474e'

# Robust ResNet 3.0 attention
model_hash_resnet_attention = 'e7e69a34ce043a1509d6909a9fc0da25'

trainer_fn = 'nnvision.training.trainers.nnvision_trainer'
trainer_hash='7eba3d5e8d426d6bbcd3f248565f8cfb'

# this will be the keys for two models
common_key = dict(dataset_hash=dataset_hash,trainer_fn=trainer_fn, trainer_hash=trainer_hash)

gauss_models = dict(model_hash=model_hash_resnet_gauss)
attn_models = dj.AndList([dict(model_hash=model_hash_resnet_attention), 'seed<11000'])

pop_key = dj.AndList([common_key, [gauss_models, attn_models]])
#pop_key = dj.AndList([common_key, gauss_models])

print(pop_key)
time.sleep(random.randint(1, 60))
# populate model table, will populate 20 entries: 10seeds*2models
# we'll later take the best 5 for each model
print('About to start training, if there is something to train')
TrainedModel().populate(pop_key, display_progress=True, reserve_jobs=True)