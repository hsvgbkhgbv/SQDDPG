"""
Auxiliary functions for building environments
"""
import os
import json
import warnings

from gym import error


def read_config(config_path, config_dict=None):
    """
    Returns the parsed information from the config file and specific info can be passed or
    overwritten with the config_dict. Full example below:
    {
        "render_options": {"depth": true},
        "pickup_put_interaction": true,
        "pickup_objects": [
            "Mug",
            "Apple",
            "Book"
        ],
        "acceptable_receptacles": [
            "CounterTop",
            "TableTop",
            "Sink"
        ],
        "openable_objects": [
            "Microwave"
        ],
        "scene_id": "FloorPlan28",
        "grayscale": true,
        "resolution": [128, 128],
        "task": {
            "task_name": "PickUp",
            "target_object": {"Mug": 1}
        }
    }
    """
    config_path = os.path.join(os.path.dirname(__file__), config_path)
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        raise error.Error('No config file found at: {}. Exiting'.format(config_path))

    if config_dict:
        for key in config_dict:
            # if key is task, need to loop through inner task obj and check for overwrites
            if key == 'task':
                for task_key in config_dict[key]:
                    if task_key in config[key]:
                        warnings.warn('Key: [\'{}\'][\'{}\'] already in config file with value {}. '
                                      'Overwriting with value: {}'.format(key, task_key,
                                                config[key][task_key], config_dict[key][task_key]))

                    config[key][task_key] = config_dict[key][task_key]
            # else just a regular check
            elif key in config:
                warnings.warn('Key: {} already in config file with value {}. '
                              'Overwriting with value: {}'.format(key, config[key],
                                                                  config_dict[key]))
                config[key] = config_dict[key]
            # key is not in config file so therefore we add it
            else:
                config[key] = config_dict[key]

    return config


class InvalidTaskParams(Exception):
    """
    Raised when the user inputs the wrong parameters for creating a task.
    """
    pass
