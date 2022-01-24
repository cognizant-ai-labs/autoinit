
# Copyright (C) 2021-2022 Cognizant Digital Business, Evolutionary AI.
# All Rights Reserved.
# Issued under the Academic Public License.
#
# You can be released from the terms, and requirements of the Academic Public
# License by purchasing a commercial license.
# Purchase of a commercial license is mandatory for any use of the
# AutoInit Software in commercial settings.
#
# END COPYRIGHT

import json
import os
from unittest import TestCase

from autoinit.initializer.initialize_weights import AutoInit
from tensorflow.keras import Model

class InitializationTest(TestCase):
    """
    Test that different kinds of models can be initialized without error.
    """

    def setUp(self):
        self.model_jsons_dir = os.path.join('tests', 'model_jsons')
        self.model_jsons = ['chestxray.json',
                            'toxicity.json',
                            'textclassifier.json',
                            'omniglot.json',
                            'mnist.json',
                            'pmlb.json',
                            'transferimageclassifier.json']

    def instantiate_model(self, path):
        with open(path, 'r') as json_file:
            json_dict = json.load(json_file)
            model_config = json_dict['interpretation']['model']['config']
            model = Model.from_config(model_config)
            return model

    def test_initialization(self):
        for model_json in self.model_jsons:
            json_path = os.path.join(self.model_jsons_dir, model_json)
            model = self.instantiate_model(json_path)
            model = AutoInit().initialize_model(model)
