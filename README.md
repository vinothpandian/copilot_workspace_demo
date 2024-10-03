# copilot_workspace_demo

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   �
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         copilot_workspace_demo and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   �
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
�   �   ├── __init__.py             <- Makes copilot_workspace_demo a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    �   ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   �   └── train.py            <- Code to train models
    │   └── train_resnet18.py   <- Code to finetune resnet18 model
    │
    └── plots.py                <- Code to create visualizations
```

--------

## How to run the training script

To finetune the resnet18 model using the provided training script, follow these steps:

1. Ensure you have the necessary dependencies installed. You can install them using the following command:
   ```
   pip install -r requirements.txt
   ```

2. Prepare your dataset and place it in the `data/train` directory. The directory structure should be as follows:
   ```
   data/
       train/
           class1/
               img1.jpg
               img2.jpg
               ...
           class2/
               img1.jpg
               img2.jpg
               ...
           ...
   ```

3. Run the training script:
   ```
   python copilot_workspace_demo/modeling/train.py
   ```

4. The trained model will be saved in the `models` directory as `resnet18_finetuned.pth`.

5. You can evaluate the model using the validation data by running the script and checking the printed metrics.
