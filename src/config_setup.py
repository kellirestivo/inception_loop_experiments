import datajoint as dj
import os


def setup_configuration():
    dj.config["enable_python_native_blobs"] = True
    dj.config["database.host"] = 'at-database3.ad.bcm.edu'
    dj.config['database.user'] = 'kelli'
    dj.config['nnfabrik.schema_name'] = "nnfabrik_toy_V4"
    os.environ['DJ_SUPPORT_FILEPATH_MANAGEMENT'] = 'TRUE'
    os.environ['TORCH_HOME'] = '/data'

    dj.config['stores'] = {
    'minio': {
        'protocol': 'file',
        'location': '/mnt/dj-stor01/',
        'stage': '/mnt/dj-stor01/'
    }
    }

    fetch_download_path = '/data/fetched_from_attach'

    return fetch_download_path

