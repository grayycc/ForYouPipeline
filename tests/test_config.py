import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import config


def test_data_dir_points_to_existing_kuairand_data():
    assert os.path.isdir(config.DATA_DIR), f'DATA_DIR does not exist: {config.DATA_DIR}'
    assert os.path.exists(os.path.join(config.DATA_DIR, 'video_features_basic_pure.csv'))
