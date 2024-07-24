from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
import pandas as pd
import numpy as np
import json
from .agent.harvest_agent import HarvestAgent
from .berry import Berry
from .harvest_exception import FileExistsException
from .harvest_exception import AgentTypeException
from .harvest_exception import OutOfBounds
from .harvest_exception import NoEmptyCells
from .harvest_exception import NumAgentsException
from .harvest_exception import NoBerriesException
from os.path import exists
from abc import abstractmethod

class HarvestModel(Model):
    def __init__(self,num_agents,max_width,max_height,max_episodes,training,write_data,write_norms,file_string=""):
        super().__init__()
        self._num_agents = num_agents
        if self._num_agents <= 0:
            raise NumAgentsException(">0", 0)
        self.end_day = 0
        self.schedule = RandomActivation(self)
        self.max_width = max_width
        self.max_height = max_height
        self.grid = MultiGrid(self.max_width, self.max_height, False)
        self._max_days = 50
        self.max_episodes = max_episodes
        self.min_expl_prob = 0.01
        self.min_epsilon = 0.01
        self._day = 1
        self.shared_replay_buffer = {"s": [], "a": [], "r": [], "s_": [], "done": []}
        self.agent_id = 0
        self.berry_id = 0
        self.num_berries = 0
        self.episode = 1
        self.start_health = 0.8
        self.file_string = file_string
        self.training = training
        self.write_data = write_data
        self._write_norms = write_norms
        self.societal_norm_emergence_threshold = 0.9
        self.emerged_norms_current = {}
        self.emerged_norms_history = {}
        if self.training:
            self.epsilon = 0.9
        else:
            self.epsilon = 0.05
        self._init_reporters()

    @abstractmethod
    def init_berries(self):
        raise NotImplementedError

    @abstractmethod
    def observe(self):
        raise NotImplementedError
    
    def init_agents(self, agent_type):
        self._living_agents = []
        for i in range(self._num_agents):
            a = HarvestAgent(i,self,agent_type,0,self.max_width,0,self.max_height,self.training,self.epsilon,shared_replay_buffer=self.shared_replay_buffer)
            self.add_agent(a)
        self._num_living_agents = len(self._living_agents)
        self.berry_id = self._num_living_agents + 1
        assert self._num_living_agents == self._num_agents, "init {self._num_living_agents} instead of {self._num_agents}"

    def add_agent(self, a):
        self.schedule.add(a)
        self.place_agent_in_allotment(a)
        if a.agent_type != "berry":
            self.agent_id += 1
            self._living_agents.append(a)
            
    def step(self):
        self.schedule.step()
        self._day += 1
        #check for dead agents
        for a in self.schedule.agents:
            if a.agent_type == "berry" and a.foraged == True:
                self._reset_berry(a, False)
            if a.agent_type != "berry":
                if self._num_living_agents == self._num_agents and a.off_grid == False:
                    self._collect_agent_data(a)
                if a.done == True and a.off_grid == False:
                    self._remove_agent(a)
        self.epsilon = self.get_mean_epsilon()
        if self._write_norms:
            self.emerged_norms = self.get_emerged_norms()
        #if exceeded max days or all agents died, reset for new episode
        if self._day >= self._max_days or self._num_living_agents <= 0:
            self.end_day = self._day
            if self._write_norms:
                self.append_norm_dictionary_to_file(self.emerged_norms, "data/results/"+self.file_string+"_emerged_norms")
            for a in self.schedule.agents:
                if a.agent_type != "berry":
                    if a.off_grid == False:
                        a.days_survived = self._day
                    if self._write_norms and self.episode % 100 == 0:
                        self.append_norm_dictionary_to_file(a.norms_model.norm_base, "dqn_results/"+self.file_string+"_agent_"+str(a.unique_id)+"_norm_base")
                    if self.training: 
                        a.save_models()
            if self.write_data:
                self._collect_model_episode_data()
            self._reset()

    def append_norm_dictionary_to_file(self, norm_dictionary, filename):
        with open(filename, "a+") as file:
            file.seek(0)
            if not file.read(1):
                file.write("\n")
            file.seek(0, 2)
            json.dump(str(self.episode), file, indent=4)
            json.dump(norm_dictionary, file, indent=4)
            file.write(",")

    def _reset(self):
        self._living_agents = []
        self.emerged_norms_current = {}
        self.emerged_norms_history = {}
        self._day = 0
        num_agents = 0
        num_berries = 0
        self.episode += 1
        self.total_episode_reward = 0
        self._clear_grid()
        for a in self.schedule.agents:
            if a.agent_type != "berry":
                self._reset_agent(a)
                num_agents += 1
            elif a.agent_type == "berry":
                self._reset_berry(a, True)
                num_berries += 1
        assert num_agents == self._num_agents, "reset "+str(num_agents)+" agents instead of "+str(self._num_agents)
        assert num_berries == self.num_berries, "reset "+str(num_berries)+" berries instead of "+str(self.num_berries)
        self._num_living_agents = self._num_agents
    
    def _reset_agent(self, agent):
        if agent.agent_type == "berry":
            raise AgentTypeException(agent.agent_type, "berry")
        agent.reset()
        self.place_agent_in_allotment(agent)
        agent.off_grid = False
        self._living_agents.append(agent)
    
    def _reset_berry(self, berry, end_of_episode):
        if berry.agent_type != "berry":
            raise AgentTypeException("berry", berry.agent_type)
        if not end_of_episode:
            self.grid.remove_agent(berry)
        berry.reset()
        self.place_agent_in_allotment(berry)

    def _init_reporters(self):
        self.agent_reporter = pd.DataFrame({"agent_id": [],
                               "episode": [],
                               "day": [],
                               "berries": [],
                               "berries_consumed": [],
                               "berries_thrown": [],
                               "health": [],
                               "days_left_to_live": [],
                               "action": [],
                               "reward": [],
                               "num_norms": []})
        if self.write_data and not self.training:
            if exists("data/results/agent_reports_"+self.file_string+".csv"):
                raise FileExistsException("data/results/agent_reports_"+self.file_string+".csv")
            self.agent_reporter.to_csv("data/results/agent_reports_"+self.file_string+".csv", mode='a')
        self.model_episode_reporter = pd.DataFrame({"episode": [], 
                            "end_day": [],
                            "epsilon": [],
                            "mean_reward": [],
                            "mean_loss": [],
                            "max_berries": [],
                            "mean_berries": [],
                            "max_berries_consumed": [],
                            "mean_berries_consumed": [],
                            "gini_berries_consumed": [],
                            "mean_berries_thrown": [],
                            "max_health": [],
                            "mean_health": [],
                            "median_health": [],
                            "variance_health": [],
                            "deceased": [],
                            "num_emerged_norms": []})
        if self.write_data:
            if exists("data/results/model_episode_reports_"+self.file_string+".csv"):
                raise FileExistsException("data/results/model_episode_reports_"+self.file_string+".csv")
            self.model_episode_reporter.to_csv("data/results/model_episode_reports_"+self.file_string+".csv", mode='a')

    def _collect_agent_data(self, agent):
        new_entry = pd.DataFrame({"agent_id": [agent.unique_id],
                               "episode": [self.episode],
                               "day": [self._day],
                               "berries": [agent.berries],
                               "berries_consumed": [agent.berries_consumed],
                               "berries_thrown": [agent.berries_thrown],
                               "health": [agent.health],
                               "days_left_to_live": [agent.days_left_to_live],
                               "action": [agent.current_action],
                               "reward": [agent.current_reward],
                               "num_norms": [len(agent.norms_module.norm_base) if self._write_norms else None]})
        self.agent_reporter = pd.concat([self.agent_reporter, new_entry], ignore_index=True)
        if self.write_data and not self.training:
            new_entry.to_csv("data/results/agent_reports_"+self.file_string+".csv", header=None, mode='a')

    def _collect_model_episode_data(self):
        row_index_list = self.agent_reporter.index[self.agent_reporter["episode"] == self.episode].tolist()
        new_entry = pd.DataFrame({"episode": [self.episode], 
                               "end_day": [self._day],
                               "epsilon": [self.epsilon],
                               "mean_reward": [self.get_mean_reward()],
                               "mean_loss": [self.get_mean_loss()],
                               "max_berries": [self.agent_reporter["berries"].iloc[row_index_list].max()],
                               "mean_berries": [self.agent_reporter["berries"].iloc[row_index_list].mean(axis=0)],
                               "max_berries_consumed": [self.agent_reporter["berries_consumed"].iloc[row_index_list].max()],
                               "mean_berries_consumed": [self.agent_reporter["berries_consumed"].iloc[row_index_list].mean(axis=0)],
                               "gini_berries_consumed": [self.get_gini_berries_consumed()],
                               "mean_berries_thrown": [self.agent_reporter["berries_thrown"].iloc[row_index_list].mean(axis=0)],
                               "max_health": [self.agent_reporter["health"].iloc[row_index_list].max()],
                               "mean_health": [self.agent_reporter["health"].iloc[row_index_list].mean(axis=0)],
                               "median_health": [self.agent_reporter["health"].iloc[row_index_list].median()],
                               "variance_health": [self.agent_reporter["health"].iloc[row_index_list].var(axis=0)],
                               "deceased": [self._num_agents - self._num_living_agents],
                               "num_emerged_norms": [len(self.emerged_norms_history) if self._write_norms else None]})
        self.model_episode_reporter = pd.concat([self.model_episode_reporter, new_entry], ignore_index=True)
        if self.write_data:
            new_entry.to_csv("data/results/model_episode_reports_"+self.file_string+".csv", header=None, mode='a')
        return new_entry

    def check_bounds(self, cell):
        if cell[0] >= 0 and cell[0] < self.max_width:
            if cell[1] >= 0 and cell[1] < self.max_height:
                return True
        return False
    
    # def get_neighbourhood(self, x, y):
    #     neighborhood = []
    #     cols = self.max_width
    #     rows = self.max_height
    #     # Define the relative offsets for the neighboring cells
    #     offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    #     # get list of tuples for neighbourhood coords
    #     for dr, dc in offsets:
    #         r, c = x + dr, y + dc
    #         if 0 <= r < rows and 0 <= c < cols:
    #             neighborhood.append((r, c))
    #     return neighborhood
    
    # def get_neighbours(self, neighbourhood):
    #     neighbours = []
    #     assert len(neighbourhood)>=0, "need more than {neighbourhood} to get neighbourhood"
    #     for cell in neighbourhood:
    #         a = self.get_agent_by_coords(cell)
    #         if a != None:
    #             neighbours += a
    #     return neighbours
    
    # def get_berries_in_cell(self, cell):
    #     #check cell is in bounds
    #     for a in self.schedule.agents:
    #         if a.type == "berry":
    #             if a.pos == cell:
    #                 return a
    #     return False
    
    # def get_agent_by_coords(self, coords):
    #     agents = []
    #     for a in self.schedule.agents:
    #         if a.type == "berry":
    #             pass
    #         elif a.off_grid == False:
    #             if a.pos == coords:
    #                 agents.append(a)
    #     return agents
    
    # def empty_cell(self, cell):
    #     for a in self.schedule.agents:
    #         if a.pos == cell:
    #             return False
    #     return True
    
    #for new agents who aren't yet on the grid
    def place_agent_in_allotment(self, agent):
        if not self.grid.exists_empty_cells:
            raise NoEmptyCells
        cell = self.get_allotment_cell(agent)
        self.grid.place_agent(agent, cell)
    
    def move_agent_to_cell(self, agent, new_pos):
        self.grid.move_agent(agent, new_pos)

    #for agents who are on the grid
    def move_agent_in_allotment(self, agent, cell=None):
        if not self.grid.exists_empty_cells:
            raise NoEmptyCells
        if cell == None:
            cell = self.get_allotment_cell(agent)
        self.grid.move_agent(agent, cell)
    
    def _clear_grid(self):
        for a in self.schedule.agents:
            if a.agent_type != "berry" and a.off_grid:
                continue
            self.grid.remove_agent(a)

    # def remove_berry(self, berry):
    #     self.schedule.remove(berry)
    #     self.grid.remove_agent(berry)
    #     self.num_berries -= 1
    
    def _remove_agent(self, agent):
        self._num_living_agents -= 1
        self.grid.remove_agent(agent)
        agent.off_grid = True
        agent.days_left_to_live = 0
        self._living_agents = [a for a in self.schedule.agents if a.agent_type != "berry" and a.off_grid == False]
        list_living = len(self._living_agents)
        assert list_living == self._num_living_agents, "length of living agents list is {list_living} and the number of living agents is {self._num_living_agents}"
    
    def new_berry(self,min_width,max_width,min_height,max_height,allocation_id=None):
        berry = Berry(self.berry_id,self,min_width,max_width,min_height,max_height,allocation_id)
        self.schedule.add(berry)
        self.berry_id += 1
        return berry
    
    def spawn_berry(self, berry, cell=None):
        if cell == None:
            self.move_agent_in_allotment(berry)
            return
        if self.check_bounds(cell):
            self.grid.place_agent(berry, cell)
            self.num_berries += 1
        else:
            raise OutOfBounds("berry", cell)
    
    def get_cell_contents(self, cell):
        return self.grid.iter_cell_list_contents(cell)

    def get_emerged_norms(self):
        emergence_count = self._num_agents * self.societal_norm_emergence_threshold
        emerged_norms = {}
        for agent in self.schedule.agents:
            if agent.agent_type != "berry":
                for norm_name, norm_value in agent.norms_module.norm_base.items():
                    if norm_name not in emerged_norms:
                        emerged_norms[norm_name] = {"score": 0,
                                                    "numerosity": 0,
                                                    "fitness": 0,
                                                    "num_instances": 0}
                    emerged_norms[norm_name]["score"] += norm_value["score"]
                    emerged_norms[norm_name]["numerosity"] += norm_value["numerosity"]
                    emerged_norms[norm_name]["fitness"] += norm_value["fitness"]
                    emerged_norms[norm_name]["num_instances"] += 1
        emerged_norms = {norm: norm_value for norm, norm_value in emerged_norms.items() if norm_value["num_instances"] >= emergence_count}
        return emerged_norms
    
    def get_allotment_cell(self, agent):
        width = np.random.randint(agent.min_width, agent.max_width)
        height = np.random.randint(agent.min_height, agent.max_height)
        return (width, height)
    
    def get_gini_berries_consumed(self):
        berries_consumed = [a.berries_consumed for a in self.schedule.agents if a.agent_type != "berry"]
        x = sorted(berries_consumed)
        s = sum(x)
        if s == 0:
            return 0
        N = self._num_agents
        B = sum(xi * (N - i) for i, xi in enumerate(x)) / (N * s)
        return 1 + (1 / N) - 2 * B
    
    def get_mean_loss(self):
        m = 0
        if self.training:
            for agent in self.schedule.agents:
                if agent.agent_type != "berry":
                    if len(agent.losses) > 1:
                        m += np.mean(agent.losses)
            if m == 0:
                return 0
            m /= self._num_agents
        return m
    
    def get_mean_reward(self):
        m = 0
        for agent in self.schedule.agents:
            if agent.agent_type != "berry":
                m += agent.total_episode_reward
        if m == 0:
            return 0
        m /= self._num_agents
        return m
    
    def get_mean_epsilon(self):
        m = 0
        for agent in self.schedule.agents:
            if agent.agent_type != "berry":
                m += agent.epsilon
        if m == 0:
            return 0
        m /= self._num_agents
        return m
    
    def get_uneaten_berries_coordinates(self, agent_id=None):
        berries_coordinates = []
        for b in self.berries:
            if b.foraged == False:# and b.marked == False: 
                if agent_id==None:
                    berries_coordinates.append(b.pos)
                else:
                    if b.allocated_agent_id == agent_id:
                        berries_coordinates.append(b.pos)
        return berries_coordinates
    
    def get_uneaten_berry_by_coords(self, coords, agent_id=None):
        for b in self.berries:
            if b.pos == coords and b.foraged == False:# and b.marked == False:
                if agent_id == None:
                    return b
                elif b.allocated_agent_id == agent_id:
                    return b
        raise NoBerriesException(coordinates=coords)

    def get_society_well_being(self, observer, include_observer):
        society_well_being = np.array([])
        for a in self.schedule.agents:
            if (a.unique_id == observer.unique_id and not include_observer) or a.agent_type == "berry":
                continue
            elif a.done == False:
                #observe agent's coords and how many days they have left
                society_well_being = np.append(society_well_being, a.days_left_to_live)
            elif a.done == True and include_observer:
                #if looking for utility measure, don't include dead agents
                continue
            else:
                #if observing, include dead agents as 0s
                society_well_being = np.append(society_well_being, 0)
        return society_well_being
    
    def get_num_agents(self):
        return self._num_agents
    
    def get_num_living_agents(self):
        assert self._num_living_agents == len(self._living_agents)
        return self._num_living_agents
    
    def get_living_agents(self):
        return self._living_agents

    def get_write_norms(self):
        return self._write_norms
    
    def get_day(self):
        return self._day

    def get_max_days(self):
        return self._max_days