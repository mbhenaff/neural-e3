import gym, pickle, time

from skimage.color import rgb2gray
from skimage.transform import resize
from baselines.homer.abstract_environment import AbstractEnvironment
from baselines.homer.rl_acid_environment import *
from baselines.homer.noise_gen import get_sylvester_hadamhard_matrix_dim


class GenerateEnvironmentWrapper(AbstractEnvironment):
    """" Wrapper class for generating environments using names and config """

    OpenAIGym, RL_ACID, GRIDWORLD = range(3)

    def __init__(self, env_name, config, bootstrap_env=None):
        """
        :param env_name: Name of the environment to create
        :param config:  Configuration to use
        :param bootstra_env: Environment used for defining
        """

        self.tolerance = 0.5
        self.env_type = None
        self.env_name = env_name
        self.config = config

        if env_name == 'MAB':
            # Mario Brother environment
            raise NotImplementedError()

        elif env_name == 'combolock':
            # Deterministic Combination Lock

            self.env_type = GenerateEnvironmentWrapper.RL_ACID
            self.thread_safe = True

            assert config["obs_dim"] == 3 * config["horizon"] + 2, "Set obs_dim to -1 in config for auto selection"
            if bootstrap_env is not None:
                self.env = bootstrap_env
            else:
                self.env = CombinationLock(horizon=config["horizon"])

            # Reach both states at a given time step with probability at least 0.5 (minus some tolerance)
            self.homing_policy_validation_fn = lambda dist, step: str((0, step)) in dist and str((1, step)) in dist and \
                                                                  dist[str((0, step))] > 50 - self.tolerance and \
                                                                  dist[str((1, step))] > 50 - self.tolerance

        elif env_name == 'stochcombolock':
            # Stochastic Combination Lock

            self.env_type = GenerateEnvironmentWrapper.RL_ACID
            self.thread_safe = True

            if config["noise"] == "bernoulli":
                self.noise_type = Environment.BERNOULLI
                assert config["obs_dim"] == 4 * config["horizon"] + 3, "Set obs_dim to -1 in config for auto selection"
            elif config["noise"] == "gaussian":
                self.noise_type = Environment.GAUSSIAN
                assert config["obs_dim"] == 3 * config["horizon"] + 3, "Set obs_dim to -1 in config for auto selection"
            else:
                raise AssertionError("Unhandled noise type %r" % self.noise_type)

            if bootstrap_env is not None:
                self.env = bootstrap_env
            else:
                self.env = StochasticCombinationLock(horizon=config["horizon"], swap=0.5, noise_type=self.noise_type)

            # Reach the two states with probability at least 0.25 each and the third state with probability at least 0.5
            self.homing_policy_validation_fn = lambda dist, step: \
                str((0, step)) in dist and str((1, step)) in dist and str((2, step)) in dist and \
                dist[str((0, step))] + dist[str((1, step))] > 50 - self.tolerance and \
                dist[str((2, step))] > 50 - self.tolerance

        elif env_name == 'diabcombolock':
            # Diabolical Stochastic Combination Lock

            self.env_type = GenerateEnvironmentWrapper.RL_ACID
            self.thread_safe = True
            self.trajectories = []
            self.trajectory_cntr = 0
            self.num_envs = 1
            

            # if config["noise"] == "bernoulli":
            #     self.noise_type = Environment.BERNOULLI
            #     assert config["obs_dim"] == 4 * config["horizon"] + 3, "Set obs_dim to -1 in config for auto selection"
            # elif config["noise"] == "gaussian":
            #     self.noise_type = Environment.GAUSSIAN
            #     assert config["obs_dim"] == 3 * config["horizon"] + 3, "Set obs_dim to -1 in config for auto selection"
            # else:
            #     raise AssertionError("Unhandled noise type %r" % self.noise_type)

            if config["noise"] == "bernoulli":

                self.noise_type = Environment.BERNOULLI
                assert config["obs_dim"] == 2 * config["horizon"] + 4, "Set obs_dim to -1 in config for auto selection"

            elif config["noise"] == "gaussian":

                self.noise_type = Environment.GAUSSIAN
                assert config["obs_dim"] == config["horizon"] + 4, "Set obs_dim to -1 in config for auto selection"

            elif config["noise"] == "hadamhard":

                self.noise_type = Environment.HADAMHARD
                assert config["obs_dim"] == get_sylvester_hadamhard_matrix_dim(config["horizon"] + 4), \
                    "Set obs_dim to -1 in config for auto selection"

            elif config["noise"] == "hadamhardg":

                self.noise_type = Environment.HADAMHARDG
                assert config["obs_dim"] == get_sylvester_hadamhard_matrix_dim(config["horizon"] + 4), \
                    "Set obs_dim to -1 in config for auto selection"

            else:
                raise AssertionError("Unhandled noise type %r" % config["noise"])

            if bootstrap_env is not None:
                self.env = bootstrap_env
            else:
                self.env = DiabolicalCombinationLock(horizon=config["horizon"], swap=0.5,
                                                     num_actions=10, anti_shaping_reward=0.1,
                                                     noise_type=self.noise_type)

            # Reach the two states with probability at least 0.25 each and the third state with probability at least 0.5
            self.homing_policy_validation_fn = lambda dist, step: \
                str((0, step)) in dist and str((1, step)) in dist and str((2, step)) in dist and \
                dist[str((0, step))] + dist[str((1, step))] > 50 - self.tolerance and \
                dist[str((2, step))] > 50 - self.tolerance

        elif env_name == 'maze':
            # Maze world

            self.env_type = GenerateEnvironmentWrapper.RL_ACID
            self.thread_safe = True

            if bootstrap_env is not None:
                self.env = bootstrap_env
            else:
                self.env = RandomGridWorld(M=3, swap=0.1, dim=2, noise=0.0)

            self.homing_policy_validation_fn = None

        elif env_name == 'montezuma':
            # Montezuma Revenge

            self.env_type = GenerateEnvironmentWrapper.OpenAIGym
            self.thread_safe = True
            self.num_repeat_action = 4  # Repeat each action these many times.

            if bootstrap_env is not None:
                self.env = bootstrap_env
            else:
                self.env = gym.make('MontezumaRevengeDeterministic-v4')

            # Since we don't have access to underline state in this problem, we cannot define a validation function
            self.homing_policy_validation_fn = None

        elif env_name == 'gridworld' or env_name == 'gridworld-feat':
            # Grid World

            self.env_type = GenerateEnvironmentWrapper.GRIDWORLD
            self.thread_safe = True

            if bootstrap_env is not None:
                self.env = bootstrap_env
            else:
                self.env = GridWorld(num_grid_row=4, num_grid_col=4, horizon=config["horizon"], obs_dim=config["obs_dim"])

            reachable_states = self.env.get_reachable_states()
            num_states = self.env.get_num_states()

            self.homing_policy_validation_fn = lambda dist, step: all(
                [str(state) in dist and dist[str(state)] >= 1.0 / float(max(1, num_states)) - self.tolerance
                 for state in reachable_states[step]])

        else:
            raise AssertionError("Environment name %r not in RL Acid Environments " % env_name)

    def generate_homing_policy_validation_fn(self):

        if self.homing_policy_validation_fn is not None:
            return self.homing_policy_validation_fn

    def step(self, action):

        if self.env_type == GenerateEnvironmentWrapper.RL_ACID:

            observation, reward, info = self.env.act(action)            
            done = self.env.h == self.config['horizon']
#            done = observation is None
            # TODO return non-none observation when observation is None.
            if self.trajectory_cntr % 100 == 0:
                self.trajectories[-1].extend([action, float(reward), info])
            return observation, float(reward), done, info

        elif self.env_type == GenerateEnvironmentWrapper.OpenAIGym:

            # Repeat the action K steps
            for _ in range(self.num_repeat_action):
                image, reward, done, info = self.env.step(action)
            image = self.openai_gym_process_image(image)
            assert "state" not in info
            info["state"] = self.openai_ram_for_state()
            return image, float(reward), done, info

        elif self.env_type == GenerateEnvironmentWrapper.GRIDWORLD:

            image, reward, done, info = self.env.step(action)
            return image, float(reward), done, info

        else:
            raise AssertionError("Unhandled environment type %r" % self.env_type)

    def reset(self):


        if self.env_type == GenerateEnvironmentWrapper.RL_ACID:

            obs = self.env.start_episode()
            '''
            if len(self.trajectories) > 100000:
                with open('./tmp/trajdump_%d.pickle' % time.time(), 'wb') as handle:
                    pickle.dump(self.trajectories, handle, protocol=pickle.HIGHEST_PROTOCOL)
                self.trajectories = []
            '''
            self.trajectory_cntr += 1
            if self.trajectory_cntr % 100 == 0:
                self.trajectories.append([obs[0], obs[1]])
            obs = obs[0]
            
            return obs

        elif self.env_type == GenerateEnvironmentWrapper.OpenAIGym:

            image = self.env.reset()
            image = self.openai_gym_process_image(image)
            return image, {"state": self.openai_ram_for_state()}

        elif self.env_type == GenerateEnvironmentWrapper.GRIDWORLD:

            return self.env.reset()

        else:
            raise AssertionError("Unhandled environment type %r" % self.env_type)

    def openai_gym_process_image(self, image):

        if self.env_name == "montezuma":
            image = image[34: 34 + 160, :160]       # 160 x 160 x 3
            image = image/256.0

            if self.config["obs_dim"] == [1, 160, 160, 3]:
                return image
            elif self.config["obs_dim"] == [1, 84, 84, 1]:
                image = resize(rgb2gray(image), (84, 84), mode='constant')
                image = np.expand_dims(image, 2)  # 84 x 84 x 1
                return image
            else:
                raise AssertionError("Unhandled configuration %r" % self.config["obs_dim"])
        else:
            raise AssertionError("Unhandled OpenAI Gym environment %r" % self.env_name)

    def openai_ram_for_state(self):
        """ Create State for OpenAI gym using RAM. This is useful for debugging. """

        ram = self.env.env._get_ram()

        if self.env_name == "montezuma":
            # x, y position and orientation of agent, x-position of the skull and position of items like key
            state = "(%d, %d, %d, %d, %d)" % (ram[42], ram[43], ram[52], ram[47], ram[67])
            return state
        else:
            raise NotImplementedError()

    def get_optimal_value(self):

        if self.env_name == 'combolock' or self.env_name == 'stochcombolock' or self.env_name == 'diabcombolock':
            return self.env.get_optimal_value()
        else:
            return None

    def is_thread_safe(self):
        return self.thread_safe

    @staticmethod
    def adapt_config_to_domain(env_name, config):
        """ This function adapts the config to the environment.
        """

        if config["obs_dim"] == -1:

            if env_name == 'combolock':
                config["obs_dim"] = 3 * config["horizon"] + 2

            elif env_name == 'stochcombolock' or env_name == 'diabcombolock':

                # if config["noise"] == "bernoulli":
                #     config["obs_dim"] = 4 * config["horizon"] + 3
                # elif config["noise"] == "gaussian":
                #     config["obs_dim"] = 3 * config["horizon"] + 3
                # else:
                #     raise AssertionError("Unhandled noise type %r" % config["noise"])

                if config["noise"] == "bernoulli":
                    config["obs_dim"] = 2 * config["horizon"] + 4
                elif config["noise"] == "gaussian":
                    config["obs_dim"] = config["horizon"] + 4
                elif config["noise"] == "hadamhard":
                    config["obs_dim"] = get_sylvester_hadamhard_matrix_dim(config["horizon"] + 4)
                elif config["noise"] == "hadamhardg":
                    config["obs_dim"] = get_sylvester_hadamhard_matrix_dim(config["horizon"] + 4)
                else:
                    raise AssertionError("Unhandled noise type %r" % config["noise"])

            else:
                raise AssertionError("Cannot adapt to unhandled environment %s" % env_name)

    def get_bootstrap_env(self):
        """ Environments which are thread safe can be bootstrapped. There are two ways to do so:
        1. Environment with internal state which can be replicated directly.
            In this case we return the internal environment.
        2. Environments without internal state which can be created exactly from their name.
            In this case we return None """

        assert self.thread_safe, "To bootstrap it must be thread safe"
        if self.env_name == 'stochcombolock' or self.env_name == 'combolock' or self.env_name == 'diabcombolock':
            return self.env
        else:
            return None

    def save_environment(self, folder_name, trial_name):

        if self.env_type == GenerateEnvironmentWrapper.RL_ACID:
            return self.env.save(folder_name + "/trial_%r_env" % trial_name)
        else:
            pass        # Nothing to save

    def load_environment_from_folder(self, env_folder_name):

        if self.env_type == GenerateEnvironmentWrapper.RL_ACID:
            self.env = self.env.load(env_folder_name)
        else:
            raise AssertionError("Cannot load environment for Non RL Acid settings")

    def is_deterministic(self):
        raise NotImplementedError()

    @staticmethod
    def make_env(env_name, config, bootstrap_env):
        return GenerateEnvironmentWrapper(env_name, config, bootstrap_env)
