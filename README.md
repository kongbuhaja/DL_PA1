# HYU-AUE8088, Understanding and Utilizing Deep Learning
## PA #1. Image Classification

# Files

```bash
├── README.md
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── AlexNet.py
│   ├── config.py
│   ├── dataset.py
│   ├── metric.py
│   ├── network.py
│   ├── util.py
│   └── ViT.py
├── test.py
└── train.py
```


# 0. Preparation

### Setup virtual environment
- Create python virtual environment
```bash
$ sh pa1.sh
```
### Wandb setup
- Login

```bash
$ wandb login
```

- Specify your Wandb entity
```bash
$ echo "export WANDB_ENTITY={YOUR_WANDB_ENTITY}" >> ~/.bashrc
$ source ~/.bashrc
```

# 1. [TODO] Evaluation metric
- Support Accuracy, F1Score(Macro, Micro)


# 2. [TODO] Train models
| Model | ViT | ResNet18 | AlexNet |  
| ------------- | ------ | ------| ------ |
| **F1Score**  | 0.3646 | 0.4689 | 0.3993 |
