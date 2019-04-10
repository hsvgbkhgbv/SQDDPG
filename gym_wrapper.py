from gym import spaces


class GymWrapper(object):

    def __init__(self, env):
        self.env = env
        self.obs_space = self.env.observation_space
        self.act_space = self.env.action_space

    def __call__(self):
        return self.env

    def get_num_of_agents(self):
        return self.env.n

    def get_shape_of_obs(self):
        obs_shapes = []
        for obs in self.obs_space:
            if isinstance(obs, spaces.Box):
                obs_shapes.append(obs.shape)
        assert len(self.obs_space) == len(obs_shapes)
        return obs_shapes

    def get_output_shape_of_act(self):
        act_shapes = []
        for act in self.act_space:
            if isinstance(act, spaces.Discrete):
                act_shapes.append(act.n)
            elif isinstance(act, spaces.MultiDiscrete):
                act_shapes.append(act.high - act.low + 1)
            elif isinstance(act, spaces.Boxes):
                assert act.shape == 1
                act_shapes.append(act.shape)
        return act_shapes

    def get_dtype_of_obs(self):
        return [obs.dtype for obs in self.obs_space]

    def get_input_shape_of_act(self):
        act_shapes = []
        for act in self.act_space:
            if isinstance(act, spaces.Discrete):
                act_shapes.append(act.n)
            elif isinstance(act, spaces.MultiDiscrete):
                act_shapes.append(act.shape)
            elif isinstance(act, spaces.Boxes):
                assert act.shape == 1
                act_shapes.append(act.shape)
        return act_shapes
