class EthicsModule():
    def __init__(self,model,unique_id,shaped_reward):
        self.model = model
        self.unique_id = unique_id
        self.shaped_reward = shaped_reward
        
    #after acting, look to see if you improved the minimum experience or not
    def maximin(self, min_days_left, min_agents, self_in_min, have_berries):
        for a in min_agents:
            #if the minimum experience was improved, positive shaped reward
            if a.days_left_to_live > min_days_left:
                return self.shaped_reward
        #else, negative
        if self_in_min == False:
            if have_berries == True:
                return -self.shaped_reward
        return 0
    
    def get_social_welfare(self):
        ordered_agents = sorted(self.model.living_agents, key=lambda x: x.days_left_to_live, reverse=True)
        min_agents = []
        self_in_min = False
        for a in ordered_agents:
            if a.days_left_to_live == ordered_agents[-1].days_left_to_live:
                min_agents.append(a)
                if a.unique_id == self.unique_id:
                    self_in_min = True
        #returns the minimum days left to live, the agents in that list, and whether you are in that list
        return ordered_agents[-1].days_left_to_live, min_agents, self_in_min

        test