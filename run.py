from src.scenarios.basic import BasicHarvest
from src.scenarios.capabilities import CapabilitiesHarvest
from src.scenarios.allotment import AllotmentHarvest
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def run_simulation(model_inst):
    while (model_inst.training and model_inst.epsilon > model_inst.min_episilon) or (not model_inst.training and model_inst.episode <= model_inst.max_episodes):
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

parser = argparse.ArgumentParser(description="Program options")
parser.add_argument("option", choices=["test", "train", "generate_graphs"],
                    help="Choose the program operation")
args = parser.parse_args()

if args.option not in ["test", "train", "generate_graphs"]:
    print("Please choose 'test', 'train', or 'generate_graphs'.")
elif args.option == "test" or args.option == "train":
    if args.option == "test":
        scenario = input("What type of scenario do you want to run (capabilities, allotment): ")
        if scenario not in ["capabilities", "allotment"]:
            print("Invalid scenario. Please choose 'capabilities', or 'allotment'.")
    else:
        scenario = "basic"
    agent_type = input("What type of agent do you want to implement (baseline, rawlsian): ")
    if agent_type not in ["baseline", "rawlsian"]:
        print("Invalid agent type. Please choose 'baseline' or 'rawlsian'.")
    write_data = input("Do you want to write data to file? (y, n): ")
    if write_data not in ["y", "n"]:
        print("Invalid choice. Please choose 'y' or 'n'.")
    if write_data == "y":
        write_data = True
        print("Data will be written into data/results.")
    elif write_data == "n":
        write_data = False
    write_norms = input("Do you want to write norms to file? (y, n): ")
    if write_norms not in ["y", "n"]:
        print("Invalid choice. Please choose 'y' or 'n'.")
    if write_norms == "y":
        write_norms = True
        print("Norms will be written into data/results.")
    elif write_norms == "n":
        write_norms = False
    if args.option == "train":
        training = True
    else:
        max_episodes = input("How many episodes do you want to run: ")
        try:
            max_episodes = int(max_episodes)
        except ValueError:
            print("Please enter an integer.")
        training = False
    file_string = scenario+"_"+agent_type
    create_and_run_model(scenario, agent_type, max_episodes, training, write_data, write_norms, file_string)
elif args.option == "generate_graphs":
  # Run your graph generation function here
  pass