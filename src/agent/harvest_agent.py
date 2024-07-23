from .dqn.dqn_agent import DQNAgent
from .moving_module import MovingModule
from .norms_module import NormsModule
from .ethics_module import EthicsModule

class HarvestAgent(DQNAgent):
    def __init__(self,unique_id,model,agent_type,min_width,max_width,min_height,max_height,n_features,training,epsilon,shared_replay_buffer=None):
        self.actions = ["north", "east", "south", "west", "throw", "eat"]
        #dqn agent class handles learning and action selection
        super().__init__(unique_id,model,agent_type,self.actions,n_features,training,epsilon,shared_replay_buffer=shared_replay_buffer)
        self.health = 0.8
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
        self.type = agent_type
        self.moving_module = MovingModule(self.unique_id, model, training, max_width, max_height)
        self.norm_model = NormsModule(model,self.unique_id)
        self.norm_clipping_frequency = 10
        if agent_type == "rawlsian":
            self.rewards = self.get_rawlsian_rewards()
            self.ethics_model = EthicsModule(model,self.unique_id,self.rewards["shaped_reward"])
        else:
            self.rewards = self.get_baseline_rewards()
        self.off_grid = False
        self.current_action = None
        
    def execute_transition(self, action):
        done = False
        self.current_action = action
        if self.model.write_norms:
            self.norm_model.update_norm_age()
            antecedent = self.norm_model.get_antecedent(self.health, self.berries)
        if self.type == "rawlsian":
            if self.berries > 0:
                have_berries = True
            else:
                have_berries = False
            min_days_left, min_agents, self_in_min = self.ethics_model.get_social_welfare()
        reward = self._perform_action(action)
        next_state = self.model.observe(self)
        #done, reward = self._update_attributes(reward) -> before or after ethics module?
        if self.type == "rawlsian":
            reward += self.ethics_model.maximin(min_days_left, min_agents, self_in_min, have_berries)
        done, reward = self.update_attributes(reward)
        if self.model.write_norms:
            self.norm_model.update_norm(antecedent, self.actions[action], reward)
            if self.model.day % self.norm_clipping_frequency == 0:
                self.norm_model.clip_norm_base()
        return reward, next_state, done
    
    def _perform_action(self, action):
        x, y = self.pos
        reward = 0
        #action 0
        if self.actions[action] == "move":
            reward = self._move()
        #action 1
        elif self.actions[action] == "eat":
            reward = self._eat()
        #action 2
        elif self.actions[action] == "throw":
            reward = self._throw()
        return reward, x, y

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
            self.model.move_agent(self, new_pos)
        return self._rewards["neutral_reward"]
    
    def _throw(self):
        if self.berries <= 0:
            return self.rewards["no_berries"]
        #have to have a minimum amount of health to throw
        if self.health < self.low_health_threshold:
            return self.rewards["insufficient_health"]
        benefactor = self._choose_benefactor()
        if not benefactor:
            return self.rewards["no_benefactor"]
        assert(benefactor.type != "berry")
        benefactor.health += self.berry_health_payoff 
        benefactor.berries_consumed += 1
        benefactor.days_left_to_live = benefactor.get_days_left_to_live()
        self.berries -= 1
        self.berries_thrown += 1
        return self.rewards["throw"]
    
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
            return self.rewards["eat"]
        else:
            return self.rewards["no_berries"]
    
    def get_baseline_rewards(self):
        rewards = {"crash": -0.2,
                   "no_berries": -0.2,
                   "no_benefactor": -0.2,
                   "insufficient_health": -0.2,
                   "empty_forage": 0,
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
                   "empty_forage": 0,
                   "shaped_reward": 0.4,
                   "throw": 0.5,
                   "forage": 0.8,
                   "eat": 0.8
                   }
        return rewards

    def get_days_left_to_live(self):
        health = self.health
        health += self.berry_health_payoff * self.berries
        days_left_to_live = health / self.health_decay
        if days_left_to_live < 0:
            return 0
        return days_left_to_live
    
    def update_attributes(self, reward):
        done = False
        self.health -= self.health_decay
        self.days_left_to_live = self.get_days_left_to_live()
        if len(self.model.living_agents) < self.model.num_agents:
            reward -= 1
            self.days_survived = self.model.day
            done = True
        if self.health <= 0:
            #environment class checks for dead agents to remove at the end of each step
            done = True
            self.days_survived = self.model.day
            self.health = 0
            reward -= 1
        if self.model.day == self.model.max_days - 1:
            reward += 1
        if self.berries > self.max_berries:
            self.max_berries = self.berries
        return done, reward