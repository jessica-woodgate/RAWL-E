from src.scenarios.basic import BasicHarvest
from src.scenarios.capabilities import CapabilitiesHarvest
from src.scenarios.allotment import AllotmentHarvest
import src.data_analysis as data_analysis
import pandas as pd
import argparse

def generate_graphs(scenario):
    path = "data/"+scenario+"/"
    files = [path+"baseline.csv",path+"rawlsian.csv"]
    labels = ["baseline", "rawlsian"]
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    data_analysis.display_violin_plot_df_list(dfs, labels, "day", "data/results/violin_end_day", "Violin Plot of Episode Length", "End Day")
    data_analysis.display_violin_plot_df_list(dfs, labels, "total_berries", "data/results/violin_total_berries", "Violin Plot of Total Berries Consumed", "Berries Consumed")


def run_simulation(model_inst):
    while (model_inst.training and model_inst.epsilon > model_inst.min_epsilon) or (not model_inst.training and model_inst.episode <= model_inst.max_episodes):
        model_inst.episode
        model_inst.step()
    num_episodes = model_inst.episode
    return num_episodes

def create_and_run_model(scenario, agent_type, max_episodes, training, write_data, write_norms, file_string):
    if agent_type == "baseline":
        num_baseline = 2
        num_rawlsian = 0
    elif agent_type == "rawlsian":
        num_baseline = 0
        num_rawlsian = 2
    if scenario == "basic":
        model_inst = BasicHarvest(num_baseline,num_rawlsian,max_episodes,training,write_data,write_norms,file_string)
    elif scenario == "capabilities":
        model_inst = CapabilitiesHarvest(num_baseline,num_rawlsian,max_episodes,training,write_data,write_norms,file_string)
    elif scenario == "allotment":
        model_inst = AllotmentHarvest(num_baseline,num_rawlsian,max_episodes,training,write_data,write_norms,file_string)
    else:
        ValueError("Unknown argument: "+scenario)
    run_simulation(model_inst)

def get_integer_input(prompt):
    while True:
        try:
            value = int(input(prompt))
            return value
        except ValueError:
            print("Invalid input. Please enter an integer.")

parser = argparse.ArgumentParser(description="Program options")
parser.add_argument("option", choices=["test", "train", "generate_graphs"],
                    help="Choose the program operation")
args = parser.parse_args()

if args.option not in ["test", "train", "generate_graphs"]:
    print("Please choose 'test', 'train', or 'generate_graphs'.")
elif args.option == "test" or args.option == "train":
    if args.option == "test":
        scenario = input("What type of scenario do you want to run (capabilities, allotment): ")
        while scenario not in ["capabilities", "allotment"]:
            scenario = input("Invalid scenario. Please choose 'capabilities', or 'allotment': ")
    else:
        scenario = "basic"
    agent_type = input("What type of agent do you want to implement (baseline, rawlsian): ")
    while agent_type not in ["baseline", "rawlsian"]:
        agent_type = input("Invalid agent type. Please choose 'baseline' or 'rawlsian': ")
    write_data = input("Do you want to write data to file? (y, n): ")
    while write_data not in ["y", "n"]:
        write_data = input("Invalid choice. Please choose 'y' or 'n': ")
    if write_data == "y":
        write_data = True
        print("Data will be written into data/results.")
    elif write_data == "n":
        write_data = False
    write_norms = input("Do you want to write norms to file? (y, n): ")
    while write_norms not in ["y", "n"]:
        write_norms = input("Invalid choice. Please choose 'y' or 'n': ")
    if write_norms == "y":
        write_norms = True
        print("Norms will be written into data/results.")
    elif write_norms == "n":
        write_norms = False
    if args.option == "train":
        training = True
        print("Model variables will be written into model_variables/current_run")
        max_episodes = 0
    else:
        max_episodes = get_integer_input("How many episodes do you want to run: ")
        training = False
    file_string = scenario+"_"+agent_type
    create_and_run_model(scenario, agent_type, max_episodes, training, write_data, write_norms, file_string)
elif args.option == "generate_graphs":
    scenario = input("What type of scenario do you want to generate graphs for (capabilities, allotment): ")
    while scenario not in ["capabilities", "allotment"]:
        scenario = input("Invalid scenario. Please choose 'capabilities', or 'allotment': ")
    print("Graphs will be saved in data/results")
    generate_graphs(scenario)