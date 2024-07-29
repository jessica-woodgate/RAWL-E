import numpy as np

class EthicsModule():
    """
    on each step, agent updates it's ability to act cooperatively (has berries; is healthy)
    agent calls update_state which tracks the measure of well-being before the agent acts
    after acting, agent calls get_sanction which calls the chosen principle to generate a sanction indicating alignment
    """
    def __init__(self,agent_id,shaped_reward):
        self.agent_id = agent_id
        self._shaped_reward = shaped_reward
        self._can_help = False
        self._current_principle = None
        self._society_well_being = None
        self._previous_min = None
        self._number_of_minimums = None

    def update_state(self, society_well_being, day, can_help):
        self.day = day
        self._can_help = can_help
        self._previous_min, self._number_of_minimums = self._maximin_welfare(society_well_being)
    
    def _maximin_welfare(self, society_well_being):
        min_value = min(society_well_being)
        count = np.count_nonzero(society_well_being==min_value)
        #print("day",self.day,"agent", self.agent_id, "maximin welfare", society_well_being, "min is", min(society_well_being), "count is", count)
        return min_value, count

    def maximin_sanction(self, society_well_being):
        current_min, current_number_of_current_mins = self._maximin_welfare(society_well_being)
        current_number_of_previous_mins = np.count_nonzero(society_well_being==self._previous_min)
        #if the global min has been made better, pos reward
        if current_min > self._previous_min:
            #print("day",self.day,"agent", self.agent_id, "current_min", current_min, "previous_min", self._previous_min, "returning pos reward", "can help", self._can_help)
            return self._shaped_reward
        #if the global min has been made worse, neg reward
        elif current_min < self._previous_min:
            #print("day",self.day,"agent", self.agent_id, "current_min", current_min, "previous_min", self._previous_min, "returning neg reward", "can help", self._can_help)
            return -self._shaped_reward
        #if the global min has not changed, but there are fewer instances of it, pos reward
        elif current_number_of_previous_mins < self._number_of_minimums and current_min == self._previous_min:
            #print("day",self.day,"agent", self.agent_id, "current_min", current_min, "previous_min", self._previous_min, "returning pos less numbers reward", "can help", self._can_help)
            return self._shaped_reward
        #if the global min has not changed, and there are more instances of it, neg reward
        elif current_number_of_previous_mins > self._number_of_minimums and current_min == self._previous_min:
            #print("day",self.day,"agent", self.agent_id, "current_min", current_min, "previous_min", self._previous_min, "returning neg more numbers reward", "can help", self._can_help)
            return -self._shaped_reward
        #print("day",self.day,"agent", self.agent_id, "current_min", current_min, "previous_min", self._previous_min, "returning neutral reward", "can help", self._can_help)
        return 0
        
    # #after acting, look to see if you improved the minimum experience or not
    # def maximin(self, min_days_left, min_agents, self_in_min, have_berries):
    #     for a in min_agents:
    #         #if the minimum experience was improved, positive shaped reward
    #         if a.days_left_to_live > min_days_left:
    #             return self.shaped_reward
    #     #else, negative
    #     if self_in_min == False:
    #         if have_berries == True:
    #             return -self.shaped_reward
    #     return 0
    
    # def get_social_welfare(self, living_agents):
    #     ordered_agents = sorted(living_agents, key=lambda x: x.days_left_to_live, reverse=True)
    #     min_agents = []
    #     self_in_min = False
    #     for a in ordered_agents:
    #         if a.days_left_to_live == ordered_agents[-1].days_left_to_live:
    #             min_agents.append(a)
    #             if a.unique_id == self.unique_id:
    #                 self_in_min = True
    #     #returns the minimum days left to live, the agents in that list, and whether you are in that list
    #     return ordered_agents[-1].days_left_to_live, min_agents, self_in_min