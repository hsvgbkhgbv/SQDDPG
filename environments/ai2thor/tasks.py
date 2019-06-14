"""
Different task implementations that can be defined inside an ai2thor environment
"""
import numpy as np

from environments.ai2thor.utils import InvalidTaskParams


class BaseTask:
    """
    Base class for other tasks to subclass and create specific reward and reset functions
    """
    def __init__(self, config):
        self.task_config = config
        self.max_episode_length = config.get('max_episode_length', 1000)
        # default reward is negative to encourage the agent to move more
        self.movement_reward = config.get('movement_reward', -0.01)
        self.step_num = 0

    def transition_reward(self, state):
        """
        Returns the reward given the corresponding information (state, dictionary with objects
        collected, distance to goal, etc.) depending on the task.
        :return: (args, kwargs) First elemnt represents the reward obtained at the step
                                Second element represents if episode finished at this step
        """
        raise NotImplementedError

    def reset(self):
        """
        :param args, kwargs: Configuration for task initialization
        :return:
        """
        raise NotImplementedError


class PickUpTask(BaseTask):
    """
    This task consists of picking up a target object. Rewards are only collected if the right
    object was added to the inventory with the action PickUp (See gym_ai2thor.envs.ai2thor_env for
    details). Because the agent can only carry 1 object at a time in its inventory, to receive
    a lot of reward one must learn to put objects down. Optimal behaviour will lead to the agent
    spamming PickupObject and PutObject near a receptacle. target_objects is a dict which contains
    the target objects which the agent gets reward for picking up and the amount of reward was the
    value
    """
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        # check that target objects are not selected as NON pickupables
        missing_objects = []
        for obj in kwargs['task']['target_objects'].keys():
            if obj not in kwargs['pickup_objects']:
                missing_objects.append(obj)
        if missing_objects:
            raise InvalidTaskParams('Error initializing PickUpTask. The objects {} are not '
                                    'pickupable!'.format(missing_objects))

        self.target_objects = kwargs['task'].get('target_objects', {'Mug': 1})
        self.prev_inventory = []

    def transition_reward(self, events):
        done = False
        reward = 0
        n = len(events)
        for e in events:
            r, d = self._single_transition_reward(e)
            reward = reward + r
            done = done or d 
        reward = reward / n
        return np.full(n, reward), done

    def _single_transition_reward(self, state):
        reward, done = self.movement_reward, False
        curr_inventory = state.metadata['inventoryObjects']
        object_picked_up = not self.prev_inventory and curr_inventory and \
                           curr_inventory[0]['objectType'] in self.target_objects

        if object_picked_up:
            # One of the Target objects has been picked up. Add reward from the specific object
            reward += self.target_objects.get(curr_inventory[0]['objectType'], 0)
            # print('{} reward collected!'.format(reward))

        if self.max_episode_length and self.step_num >= self.max_episode_length:
            # print('Reached maximum episode length: {}'.format(self.step_num))
            done = True

        self.prev_inventory = state.metadata['inventoryObjects']
        return reward, done

    def reset(self):
        # TODO: maybe a problem here
        self.prev_inventory = []
        self.step_num = 0
