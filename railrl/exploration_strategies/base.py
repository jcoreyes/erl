import abc

from railrl.policies.base import ExplorationPolicy


class ExplorationStrategy(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_action(self, t, observation, policy, **kwargs):
        pass

    @abc.abstractmethod
    def get_actions(self, t, observation, policy, **kwargs):
        pass

    def reset(self):
        pass


class RawExplorationStrategy(ExplorationStrategy, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_action_from_raw_action(self, action, **kwargs):
        pass

    def get_actions_from_raw_actions(self, actions, **kwargs):
        raise NotImplementedError()

    def get_action(self, t, policy, *args, **kwargs):
        action, agent_info = policy.get_action(*args, **kwargs)
        return self.get_action_from_raw_action(action, t=t), agent_info

    def get_actions(self, t, observation, policy, **kwargs):
        actions = policy.get_actions(observation)
        return self.get_actions_from_raw_actions(actions, **kwargs)

    def reset(self):
        pass


class PolicyWrappedWithExplorationStrategy(ExplorationPolicy):
    def __init__(
            self,
            exploration_strategy: ExplorationStrategy,
            policy,
    ):
        self.es = exploration_strategy
        self.policy = policy
        self.t = 0

    def set_num_steps_total(self, t):
        self.t = t

    def get_action(self, *args, **kwargs):
        return self.es.get_action(self.t, self.policy, *args, **kwargs)

    def get_actions(self, *args, **kwargs):
        return self.es.get_actions(self.t, self.policy, *args, **kwargs)

    def reset(self):
        self.es.reset()
        self.policy.reset()

    def get_param_values(self):
        return self.policy.get_param_values()

    def set_param_values(self, param_values):
        self.policy.set_param_values(param_values)

    def get_param_values_np(self):
        return self.policy.get_param_values_np()

    def set_param_values_np(self, param_values):
        self.policy.set_param_values_np(param_values)

    def to(self, device):
        self.policy.to(device)


# class PolicyWrappedWithSearch(ExplorationPolicy):
#     def __init__(
#             self,
#             policy,
#             qf1,
#             qf2,
#             search_buffer,
#             max_dist,
#     ):
#         self.policy = policy
#         self.qf1 = qf1
#         self.qf2 = qf1
#         self.search_buffer = search_buffer
#         self.state_size = search_buffer.shape[1]
#         self.max_dist = max_dist
#         self.t = 0

#     def get_distance(self, observation):
#         action = self.policy.get_actions(observation)
#         distance = torch.min(
#             self.qf1(observation, action),
#             self.qf2(observation, action),
#         )

#     def get_buffer_distance(self, i, j):


#     def get_pairwise_distance(self, start_array, goal_array=None, masked=True):
#         if goal_array == None:
#             goal_array = start_array
#         dist_matrix = []

#         for obs_index in range(start_tensor.shape[0]):
#             obs = start_tensor[obs_index]
#             obs_repeat_array = np.ones_like(goal_tensor) * np.expand_dims(obs, 0)
#             obs_goal_array = np.cat([obs_repeat_tensor, goal_array])
#             dist = self.get_distance(obs_goal_array)
#             dist_matrix.append(dist)

#         pairwise_dist = np.stack(dist_matrix)

#     # if aggregate is None:
#     #   pairwise_dist = tf.transpose(pairwise_dist, perm=[1, 0, 2])

#         mask = (pairwise_dist > self.max_dist)
#         return np.where(mask, tf.fill(pairwise_dist.shape, np.inf), 
#                         pairwise_dist)
#     else:
#       return pairwise_dist


#     def initialize_graph(self):
#         self.graph = nx.DiGraph()
#         for i in range(self.state_size):
#             for j in range(self.state_size):
#                 obs = np.stack([self.search_buffer[i], self.search_buffer[j]])
#                 dist = self.get_distance(obs)
#                 if dist < self.max_dist:
#                     self.graph.add_edge(i, j, weight=dist)

#     def get_waypoint(self, observation):
#         start = observation[:self.state_size]
#         goal = observation[self.state_size:]
#         return None


#     def set_num_steps_total(self, t):
#         self.t = t

#     def get_action(self, *args, **kwargs):


#         waypoint = self.get_waypoint(observation)
#         return self.policy.get_action()

#     def get_actions(self, *args, **kwargs):
#         return self.es.get_actions(self.t, self.policy, *args, **kwargs)

#     def reset(self):
#         self.policy.reset()

#     def get_param_values(self):
#         return self.policy.get_param_values()

#     def set_param_values(self, param_values):
#         self.policy.set_param_values(param_values)

#     def get_param_values_np(self):
#         return self.policy.get_param_values_np()

#     def set_param_values_np(self, param_values):
#         self.policy.set_param_values_np(param_values)

#     def to(self, device):
#         self.policy.to(device)