import os
import time
import random
time.sleep(random.randint(1, 10))
fetch_download_path = '/data/fetched_from_attach'
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
from nnvision.tables.scores import TestCorrelationScore, CorrelationToAverageScore
from nnvision.datasets.utility import get_validation_split, ImageCache, get_cached_loader, get_fraction_of_training_images, get_crop_from_stimulus_location

dj.config['stores'] = {
  'minio': {
    'protocol': 'file',
    'location': '/mnt/dj-stor01/',
    'stage': '/mnt/dj-stor01/'
  }
} 

print(dj.config)
numSeeds = 10
#### ADD DATASET HASH HERE ######
dataset_hash = '072e4345d9a54ad5f487f6d632ba695b'
#####################################

#Define unique hashes for models, trainer, MEI generation method
model_hash_resnet_gauss = 'ade1c26ff74aef5479499079a219474e'
model_hash_resnet_attention = 'e7e69a34ce043a1509d6909a9fc0da25'
trainer_fn = 'nnvision.training.trainers.nnvision_trainer'

trainer_hash='7eba3d5e8d426d6bbcd3f248565f8cfb'

method_hash = "a9277c97184765592390fc3bd0ed5cda"

common_key = dict(dataset_hash=dataset_hash,
                  trainer_fn=trainer_fn, 
                  trainer_hash=trainer_hash)

gauss_models = dict(model_hash=model_hash_resnet_gauss)
attn_models = dj.AndList([dict(model_hash=model_hash_resnet_attention), 'seed<11000'])

pop_key = dj.AndList([common_key, [gauss_models, attn_models]])

resnet_gauss_generator_keys = (TrainedModel() & common_key & dict(model_hash=model_hash_resnet_gauss) & 'seed<11000').fetch("KEY", limit=5, order_by="score DESC")
resnet_gauss_validator_keys = (TrainedModel() & common_key & dict(model_hash=model_hash_resnet_gauss) & 'seed>10000').fetch("KEY", limit=5, order_by="score DESC")

resnet_att_keys = (TrainedModel() & common_key & dict(model_hash=model_hash_resnet_attention)).fetch("KEY", limit=5, order_by="score DESC")
gaussValidatorEnsemble = (Ensemble.Member & resnet_gauss_validator_keys).fetch("ensemble_hash")[0]
gaussGeneratorEnsemble = (Ensemble.Member & resnet_gauss_generator_keys).fetch("ensemble_hash")[0]
attentionEnsemble = (Ensemble.Member & resnet_att_keys).fetch("ensemble_hash")[0]

unit_restriction = ((Recording.Units * CorrelationToAverageScore.Units) & pop_key)

#Restrict generation to only 6 best predicted units
selected_indices, selected_ids = unit_restriction.fetch("unit_index", "unit_id", order_by="unit_avg_correlation DESC", limit=6)
selected_keys = unit_restriction.fetch("data_key", "unit_id", "dataset_hash", order_by="unit_avg_correlation DESC", as_dict=True, limit=6)
mei_keys = dj.AndList([selected_keys, dict(method_hash=method_hash),
                       f"mei_seed <={numSeeds*10}", 'ensemble_hash!=' + '"' + str(gaussValidatorEnsemble) + '"'])

#Populate MEI table with MEIs for each of the 6 units
MEI().populate(mei_keys, reserve_jobs=True, display_progress=True)