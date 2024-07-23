from .dqn.dqn_agent import DQNAgent
from .moving_module import MovingModule
from .norms_module import NormsModule
from .ethics_module import EthicsModule
import numpy as np

class HarvestAgent(DQNAgent):
    def __init__(self,unique_id,model,agent_type,min_width,max_width,min_height,max_height,training,epsilon,shared_replay_buffer=None):
        self.actions = self._generate_actions(unique_id, model.get_num_agents())
        #dqn agent class handles learning and action selection
        super().__init__(unique_id,model,agent_type,self.actions,training,epsilon,shared_replay_buffer=shared_replay_buffer)
        self._start_health = 0.8
        self.health = self._start_health
        self.berries = 0
        self.berries_consumed = 0
        self.berries_thrown = 0
        self.days_survived = 0
        self.max_berries = 0
        self.max_width = max_width
        self.min_width = min_width
        self.width = max_width - min_width + 1
        self.max_height = max_height
        self.min_height = min_height
        self.height = max_height - min_height + 1
        self.health_decay = 0.1
        self.days_left_to_live = self.health/0.1
        self.berry_health_payoff = 0.6
        self.low_health_threshold = 0.6
        self.agent_type = agent_type
        self.moving_module = MovingModule(self.unique_id, model, training, max_width, max_height)
        self.norms_module = NormsModule(model,self.unique_id)
        self.norm_clipping_frequency = 10
        if agent_type == "rawlsian":
            self._rewards = self.get_rawlsian_rewards()
            self.ethics_model = EthicsModule(model,self.unique_id,self._rewards["shaped_reward"])
        else:
            self._rewards = self.get_baseline_rewards()
        self.off_grid = False
        self.current_action = None
        
    def execute_transition(self, action):
        done = False
        self.current_action = action
        if self.model.get_write_norms():
            self.norms_module.update_norm_age()
            antecedent = self.norms_module.get_antecedent(self.health, self.berries)
        if self.agent_type == "rawlsian":
            if self.berries > 0:
                have_berries = True
            else:
                have_berries = False
            min_days_left, min_agents, self_in_min = self.ethics_model.get_social_welfare()
        reward = self._perform_action(action)
        next_state = self.observe()
        #done, reward = self._update_attributes(reward) -> before or after ethics module?
        if self.agent_type == "rawlsian":
            reward += self.ethics_model.maximin(min_days_left, min_agents, self_in_min, have_berries)
        done, reward = self._update_attributes(reward)
        if self.model.get_write_norms():
            self.norms_module.update_norm(antecedent, self.actions[action], reward)
            if self.model.get_day() % self.norm_clipping_frequency == 0:
                self.norms_module.clip_norm_base()
        return reward, next_state, done

    #agents can see their attributes,distance to nearest berry,well being of other agents
    def observe(self):
        distance_to_berry = self.moving_module.get_distance_to_berry()
        observer_features = np.array([self.health, self.berries, self.days_left_to_live, distance_to_berry])
        agent_well_being = self.model.get_society_well_being(self, False)
        observation = np.append(observer_features, agent_well_being)
        assert len(observation) == self.n_features
        return observation
    
    def get_n_features(self):
        #agent health, berries, days left to live, distance to berry
        n_features = 4
        #feature for each observer well being
        n_features += self.model.get_num_agents() -1
        return n_features
      
    def reset(self):
        self.done = False
        self.total_episode_reward = 0
        self.berries = 0
        self.berries_consumed = 0
        self.berries_thrown = 0
        self.max_berries = 0
        self.health = self._start_health
        self.current_reward = 0
        self.days_left_to_live = self.get_days_left_to_live()
        self.days_survived = 0
        self.norms_module.norm_base  = {}
        self.moving_module.reset()

    def get_days_left_to_live(self):
        health = self.health
        health += self.berry_health_payoff * self.berries
        days_left_to_live = health / self.health_decay
        if days_left_to_live < 0:
            return 0
        return days_left_to_live

    def _generate_actions(self, unique_id, num_agents):
        """
        Generates a list of all possible actions for the agents.
        If there are lots of agents, should reconsider this function
        Args:
            num_agents: Number of agents in the environment.

        Returns:
            A list of actions, where each action is a string representing 
            "move", "eat", or "throw_AGENT_ID" (e.g., "throw_1").
        """
        actions = ["move", "eat"]
        for agent_id in range(num_agents):
            if agent_id != unique_id:
                actions.append(f"throw_{agent_id}")
        return actions
    
    def _perform_action(self, action_index):
        reward = 0
        action = self.actions[action_index]
        #action 0
        if action == "move":
            reward = self._move()
        #action 1
        elif action == "eat":
            reward = self._eat()
        #action 2+ (throw)
        else:
            agent_id = int(action.split("_")[1])
            reward = self._throw(agent_id)
        return reward

    def _move(self):
        if not self.moving_module.check_nearest_berry(self.pos):
            #if no berries have been found to walk towards, have to wait
            return self._rewards["neutral_reward"]
        #otherwise, we have a path, move towards the berry; returns True if we are at the end of the path and find a berry
        berry_found, new_pos = self.moving_module.move_towards_berry(self.pos)
        if berry_found:
            self.berries += 1
            return self._rewards["forage"]
        if new_pos != self.pos:
            self.model.move_agent_to_cell(self, new_pos)
        return self._rewards["neutral_reward"]
    
    def _throw(self, benefactor_id):
        """
        checks if it is feasible to throw a berry (have berries and have health)
        gets the agent with the matching id to the throw action
        benefactor immediately eats the berry
        """
        if self.berries <= 0:
            return self._rewards["no_berries"]
        #have to have a minimum amount of health to throw
        if self.health < self.low_health_threshold:
            return self._rewards["insufficient_health"]
        #print("trying to throw to", benefactor_id, "berries", self.berries)
        for a in self.model.get_living_agents():
            if a.unique_id == benefactor_id:
                #print("throwing to", a.unique_id)
                assert(a.agent_type != "berry")
                a.health += self.berry_health_payoff 
                a.berries_consumed += 1
                a.days_left_to_live = a.get_days_left_to_live()
                self.berries -= 1
                self.berries_thrown += 1
                return self._rewards["throw"]
        return self._rewards["no_benefactor"]
    
    def _choose_benefactor(self):
        benefactor = [a for a in self.model.living_agents if a.unique_id != self.unique_id]
        if len(benefactor) > 0:
            return benefactor[0]
        else:
            return False
    
    def _eat(self):
        if self.berries > 0:
            self.health += self.berry_health_payoff
            self.berries -= 1
            self.berries_consumed += 1
            return self._rewards["eat"]
        else:
            return self._rewards["no_berries"]
    
    def get_baseline_rewards(self):
        rewards = {"crash": -0.2,
                   "no_berries": -0.2,
                   "no_benefactor": -0.2,
                   "insufficient_health": -0.2,
                   "neutral_reward": 0,
                   "throw": 0.5,
                   "forage": 1,
                   "eat": 1
                   }
        return rewards
    
    def get_rawlsian_rewards(self):
        rewards = {"crash": -0.1,
                   "no_berries": -0.1,
                   "no_benefactor": -0.1,
                   "insufficient_health": -0.1,
                   "neutral_reward": 0,
                   "shaped_reward": 0.4,
                   "throw": 0.5,
                   "forage": 0.8,
                   "eat": 0.8
                   }
        return rewards
    
    def _update_attributes(self, reward):
        done = False
        self.health -= self.health_decay
        self.days_left_to_live = self.get_days_left_to_live()
        day = self.model.get_day()
        if len(self.model.get_living_agents()) < self.model.get_num_agents():
            reward -= 1
            self.days_survived = day
            done = True
        if self.health <= 0:
            #environment class checks for dead agents to remove at the end of each step
            done = True
            self.days_survived = day
            self.health = 0
            reward -= 1
        if day == self.model.get_max_days() - 1:
            reward += 1
        if self.berries > self.max_berries:
            self.max_berries = self.berries
        return done, reward