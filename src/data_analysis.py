import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

class DataAnalysis:
    def __init__(self, num_agents):
        self.num_agents = num_agents
    
    def mean_step_across_episodes(self, df):
        df = df.drop(["episode", "action"], axis=1)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        mean_df = df.groupby(["day", "agent_id"]).mean().reset_index()
        return mean_df
    
    def sum_step_across_episodes(self, df):
        df = df.drop(["episode", "action"], axis=1)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        sum_df = df.groupby(["day", "agent_id"]).sum().reset_index()
        return sum_df

    def normalised_sum_step_across_episodes(self, df):
        df = df.drop(["episode", "action"], axis=1)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        count_df = df.groupby(["day", "agent_id"]).size().reset_index(name="count")
        sum_df = df.groupby(["day", "agent_id"]).sum().reset_index()
        sum_df = sum_df.reset_index(drop=True)
        count_df = count_df.reset_index(drop=True)
        to_divide_columns = list(sum_df.columns)
        to_divide_columns.remove("day")
        to_divide_columns.remove("agent_id")
        sum_df.loc[:, to_divide_columns] = sum_df.loc[:, to_divide_columns].divide(count_df["count"], axis=0)
        sum_df["count"] = count_df["count"]
        return sum_df
    
    def get_num_column_value(self, df_list, column, column_value):
        count_list = []
        for df in df_list:
            count = df[column].value_counts().get(column_value, 0)
            count_list.append(count)
        return count_list

    def apply_function_to_list(self, list, function):
        results_list = []
        for item in list:
            result = function(item)
            results_list.append(result)
        return results_list

    def calculate_column_across_episode(self, df_list, df_labels, column, calculation):
        data = []
        for df in df_list:
            series = df.groupby("day")[column].apply(calculation)
            data.append(series)
        df = pd.DataFrame(data).T
        df.columns = df_labels
        return df
    
    def calculate_max(self, series):
        return series.max()

    def calculate_min(self, series):
        return series.min()

    def calculate_variance(self, series):
        return series.var()

    def calculate_gini(self, series):
        x = sorted(series)
        s = sum(x)
        if s == 0:
            return 0
        N = self.num_agents
        B = sum(xi * (N - i) for i, xi in enumerate(x)) / (N * s)
        return 1 + (1 / N) - 2 * B
    
    def calculate_total(self, series):
        return series.sum()
    
    def display_dataframe(self, df, title, y_label, save, filename):
        sns.set_palette("colorblind")
        ax = sns.lineplot(data=df)
        ax.set_xlabel("day")
        ax.set_ylabel(y_label)
        ax.legend(title="Societies", loc="upper left")
        plt.title(title)
        plt.show()
        if save:
            plt.savefig(str(filename).split()[0])
        plt.close()

    def display_violin_plot(self, df_list, df_labels, column, filename, title, y_label):
        fig, ax = plt.subplots()
        combined_df = pd.concat([df.assign(label=label) for df, label in zip(df_list, df_labels)])
        colors = sns.color_palette("colorblind", n_colors=len(df_labels))
        sns.violinplot(
            data=combined_df,
            x="label",
            y=column,
            hue="label",
            palette=colors,
            ax=ax,
            legend=False,
        )
        plt.xlabel("Society")
        plt.ylabel(y_label)
        plt.title(title)
        plt.show()
        plt.savefig(str(filename).split()[0])

    def display_violin_episode_gini(self, df_list, df_labels, column, write, filename):
        data = []
        for df in df_list:
            series = df.groupby("episode")[column].apply(self.calculate_gini)
            data.append(series)
        df = pd.DataFrame(data).T
        if write:
            df.to_csv("dqn_results/gini_index.csv")
        df.columns = df_labels
        fig, ax = plt.subplots()
        sns.violinplot(data=df, ax=ax)
        plt.xlabel(column)
        plt.ylabel('Society')
        plt.title('Violin Plot of Gini for ' + column + ' by Society')
        plt.show()
        plt.savefig(str(filename).split()[0])
    
    def days_left_to_live_results(self, sum_df_list, df_labels, write, filename):
        max_days_left_to_live = self.calculate_column_across_episode(sum_df_list, df_labels, "days_left_to_live", self.calculate_max)
        self.display_dataframe(max_days_left_to_live, "Max Days Left To Live", "Days Left To Live", filename+"_max")
        min_days_left_to_live = self.calculate_column_across_episode(sum_df_list, df_labels, "days_left_to_live", self.calculate_min)
        self.display_dataframe(min_days_left_to_live, "Min Days Left To Live", "Days Left To Live", filename+"_min")
        var_days_left_to_live = self.calculate_column_across_episode(sum_df_list, df_labels, "days_left_to_live", self.calculate_variance)
        self.display_dataframe(var_days_left_to_live, "Variance Days Left To Live", "Days Left To Live", filename+"_var")
        total_days_left_to_live = self.calculate_column_across_episode(sum_df_list, df_labels, "days_left_to_live", self.calculate_total)
        self.display_dataframe(total_days_left_to_live, "Total Days Left To Live", "Days Left To Live", filename+"_total")
        gini_days_left_to_live = self.calculate_column_across_episode(sum_df_list, df_labels, "days_left_to_live", self.calculate_gini)
        self.display_dataframe(gini_days_left_to_live, "Gini Index of Days Left To Live", "Days Left To Live", filename+"_gini")
        if write:
            max_days_left_to_live.to_csv("results/max_days_left_to_live.csv")
            min_days_left_to_live.to_csv("results/min_days_left_to_live.csv")
            var_days_left_to_live.to_csv("results/var_days_left_to_live.csv")
            gini_days_left_to_live.to_csv("results/gini_days_left_to_live.csv")
            total_days_left_to_live.to_csv("results/total_days_left_to_live.csv")

    def process_and_write_episode_dataframes(self, dataframes, df_labels):
        processed_dfs = []
        for i, df in enumerate(dataframes):
            grouped_df = df.groupby("episode")
            episode_dfs = []
            for episode, group_df in grouped_df:
                last_two_rows = group_df.tail(2).loc[:, ~group_df.columns.str.contains('^Unnamed')]
                last_two_rows["total_berries"] = last_two_rows["berries"] + last_two_rows["berries_consumed"]
                episode_dfs.append(last_two_rows)
            processed_df = pd.concat(episode_dfs)
            processed_dfs.append(processed_df)
            processed_df.to_csv(f"dqn_results/processed_episode_df_{df_labels[i]}.csv", index=False)
        return processed_dfs

    def process_and_write_agent_dataframes(self, dataframes, df_labels, max_steps):
        processed_dfs = []
        for i, df in enumerate(dataframes):
            grouped_df = df.groupby("episode")
            episode_dfs = []
            for episode, group_df in grouped_df:
                last_two_rows = group_df.tail(2)
                if not last_two_rows["day"].eq(max_steps).all():
                    new_df = group_df.iloc[:-1]
                else:
                    new_df = group_df
                episode_dfs.append(new_df)
            processed_df = pd.concat(episode_dfs)
            processed_dfs.append(processed_df)
            processed_df.to_csv(f"dqn_results/processed_agent_df_{df_labels[i]}.csv", index=False)
        return processed_dfs
    
    def create_norms_dataframe(self, filename, write, write_name=""):
        norms_data = []
        with open(filename, 'r') as file:
            data = json.load(file)
        for episode, rules in data.items():
            for rule in rules:
                attributes = rule.popitem()[1]
                attributes['episode'] = episode
                norms_data.append(attributes)
        df = pd.DataFrame(norms_data)
        if write:
            df.to_csv(write_name+".csv")
        return df