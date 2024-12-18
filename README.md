# RAWL-E
This repository includes the codebase for the AAAI 2025 paper "Operationalising Rawlsian Ethics for Fairness in Norm-Learning Agents."

Jessica Woodgate, Paul Marshall, and Nirav Ajmeri. 2025. Operationalising Rawlsian Ethics for Fairness in Norm-Learning Agents. In *Proceedings of the 39th Annual AAAI Conference on Artificial Intelligence (AAAI)*, Philadelphia, 1–9.


## Table of Contents
- [Introduction](#introduction)
- [Initialisation](#initialisation)
- [Usage](#usage)
- [Citation](#citation)

## Introduction
RAWL-E is a method for developing ethical norm-learning agents that implement the maximin principle from Rawlsian ethics in their decision-making processes. This codebase facilitates the creation of agents that promote ethical norms by balancing societal well-being with individual goals. Evaluations in simulated harvesting scenarios demonstrate that RAWL-E agents enhance social welfare, fairness, and robustness, resulting in improved minimum experiences compared to agents that do not utilise Rawlsian ethics.

## Initialisation
To set up the environment, create a virtual environment using:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate renv
```

## Usage

To run the code, use the following command:

```bash
python run.py [train] [test] [graphs]
```

## Arguments

The following arguments can be used with the `run.py` script:

- `train`: Train the norm-learning agent.
- `test`: Evaluate the performance of the trained agent.
- `graphs`: Generate relevant plots for analysis.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{Woodgate+AAAI25-Rawle,
  author = {Jessica Woodgate and Paul Marshall and Nirav Ajmeri},
  title = {Operationalising Rawlsian Ethics for Fairness in Norm-Learning Agents},
  booktitle = {Proceedings of the 39th Annual {AAAI} Conference on Artificial Intelligence ({AAAI})},
  pages = {1--9},
  month = feb,     
  year = 2025,
  publisher = {AAAI},
  address = {Philadelphia}
}
