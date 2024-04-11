# Human Robot Collaboration with Meta and Reinforcement Learning

## Project Overview
This project focuses on enhancing Human-Robot Collaboration (HRC) in construction environments, specifically during brick-laying tasks with a KUKA robot. Previous studies have identified the range of emotions experienced by construction workers when interacting with robots. Recognizing the variability in individual emotional responses, we aim to create personalized models to better adapt robot behavior to human emotions. To achieve this, we used Meta Learning techniques to determine initial model weights based on a small sample of personalized data, allowing for more effective and empathetic human-robot interactions. We also leveraged a Reinforcement Learning (RL) technique to be adapted within diverse states between workers and robots.

## Usage
The project consists of two main components: model.py and run.ipynb.

`model.py`
This script contains the implementation of the regression model that correlates robot states with human emotions. Additionally, it includes the Model-Agnostic Meta-Learning (MAML) approach for meta-learning, which is utilized to fine-tune the model for personalized emotion response prediction. It also includes the Proximal Policy Optimization (PPO) considering for continous action spaces.

To utilize the model and perform meta-learning and RL, you can import the classes and functions from this script into your own Python scripts or interactive notebooks.

`run.ipynb`
run.ipynb is a Jupyter notebook that demonstrates the analysis of the results obtained from the personalized models. It walks you through the process of loading data, applying the meta-learning and reinforcement learning model, and interpreting the results to understand how different initial weights affect the model's performance on individual emotion prediction.

## Team Members
- [Francis Baek] - Project Lead
- [Leyang Wen](https://github.com/LeyangWen) - Reinforcement Learning Part 
- [Gunwoo Yong](https://github.com/gwyong) - Reinforcement Learning, Meta Learning Part

## Publications
This project has contributed to the following publications, which provide in-depth analysis and findings related to Human-Robot Collaboration and the application of Meta Learning in personalizing robot behaviors based on human emotional responses:
- Will be updated later.

## Contributing
Contributions to this project are welcome. Please feel free to fork the repository, make changes, and submit pull requests. If you encounter any issues or have suggestions for improvement, please open an issue in the repository.

## License
Will be updated later.