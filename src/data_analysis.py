import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

num_agents = 2

def display_violin_plot_df_list(df_list, df_labels, column, filename, title, y_label):
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