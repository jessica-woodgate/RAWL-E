import json
import pandas as pd

class NormProcessing():
    def __init__(self):
        self.min_instances = 1
        self.min_fitness = 0.1
        self.min_reward = 50
    
    def proccess_norms(self, input_file, output_file):
        f = open(input_file)
        data = json.load(f)
        cooperative_data = self._count_cooperative_norms(data, output_file)
        data = self._merge_norms(data, output_file)
        return cooperative_data
    
    def _count_cooperative_norms(self, data, output_file):
        cooperative_norms = []
        n_norms = 0
        for episode_number, episode_norms in data.items():
            for norm in episode_norms:
                n_norms += 1
                norm_name = list(norm.keys())[0]
                norm_value = list(norm.values())[0]
                consequent = norm_name.split("THEN")[1].strip(",")
                if consequent == "throw":
                    norm_data = {"reward": norm_value["reward"], "numerosity": norm_value["numerosity"], "fitness": norm_value["fitness"]}
                    cooperative_norms.append(norm_data)
        print("n_norms", n_norms, "cooperative norms", len(cooperative_norms))
        print("Proportion of cooperative norms for "+output_file+" is "+str((len(cooperative_norms)/n_norms)*100))
        df = pd.DataFrame(cooperative_norms)
        df.to_csv(output_file+"_cooperative_data.csv")
        return df

    def _merge_norms(self,data,output_file):
        """
        Merges duplicates of norms into one dictionary

        Args:
            data: Norm base to remove duplicates from (dictionary).
            filename: The file to write the unique set of norms to.
            filter: Whether to filter the norms by fitness and number of instances.
            min_instances: Minimum number of instances of a norm to include in unique set.
            min_fitness: Minimum fitness of norm to include in unique set.

        Returns:
            A dictionary containing the unique set of norms.
        """
        filename = output_file+"_merged.txt"
        emerged_norms = {}
        for episode_number, episode_norms in data.items():
            #key is the episode number; value is the emerged norms from that episode
            for norm in episode_norms:
                for norm_name, norm_data in norm.items():
                    if norm_name not in emerged_norms.keys():
                        emerged_norms[norm_name] = {"reward": norm_data["reward"],
                                                    "numerosity": norm_data["numerosity"],
                                                    "fitness": norm_data["fitness"],
                                                    "adoption": norm_data["adoption"],
                                                    "num_instances_across_episodes": 1}
                    else:
                        emerged_norms[norm_name]["reward"] += norm_data["reward"]
                        emerged_norms[norm_name]["numerosity"] += norm_data["numerosity"]
                        emerged_norms[norm_name]["fitness"] += norm_data["fitness"]
                        emerged_norms[norm_name]["adoption"] += norm_data["adoption"]
                        emerged_norms[norm_name]["num_instances_across_episodes"] += 1
        emerged_norms = dict(sorted(emerged_norms.items(), key=lambda item: item[1]["fitness"], reverse=True))
        with open(filename, "a+") as file:
            file.seek(0)
            if not file.read(1):
                file.write("\n")
            file.seek(0, 2)
            json.dump(emerged_norms, file, indent=4)
            file.write("\n")
        with open(output_file+"_merged_keys.txt", "w") as keys_file:
            keys_file.write("\n".join([key for key in emerged_norms.keys()]))
        with open(output_file+"_merged_cooperative_keys.txt", "w") as keys_file:
            for key in emerged_norms.keys():
                if "throw" in key:
                    keys_file.write(key+"\n")
        return emerged_norms