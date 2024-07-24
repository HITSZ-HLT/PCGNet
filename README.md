# A Persona-Infused Cross-Task Graph Network for Multimodal Emotion Recognition with Emotion Shift Detection in Conversations

> The official implementation for paper: [*A Persona-Infused Cross-Task Graph Network for Multimodal Emotion Recognition with Emotion Shift Detection in Conversations*](https://dl.acm.org/doi/10.1145/3626772.3657944), SIGIR 2024.

<img src="https://img.shields.io/badge/Venue-SIGIR--24-blue" alt="venue"/> <img src="https://img.shields.io/badge/Status-Accepted-success" alt="status"/> <img src="https://img.shields.io/badge/Issues-Welcome-red">


## Requirements
* Python 3.10.13
* PyTorch 1.13.1
* torch_geometric 2.4.0
* torch-scatter 2.1.0
* torch-sparse 0.5.15
* CUDA 11.7

## Preparation

1. Download  [**multimodal-features**](https://www.dropbox.com/scl/fo/veblbniqjrp3iv3fs3z6p/AEzkNgWqPHHzldBZ0zEzr2Y?rlkey=yhlr653c0vnvaf1krpdkla36u&e=1&dl=0) 
2. Save data/iemocap/iemocap_features_roberta.pkl, data/iemocap/IEMOCAP_features.pkl in `data/iemocap/`; Save meld_features_roberta.pkl, data/meld/MELD_features_raw1.pkl in `data/meld/`. 


## Training & Evaluation

1. train PCGNet on IEMOCAP for ERC task
```shell
python code/run_train_erc.py --dataset IEMOCAP --data_dir data/iemocap/IEMOCAP_features.pkl \
  --valid_rate 0.0 --modals avl --lr 0.0001 --batch-size 32 --l2 0.0001 --dropout 0.2 --gamma 0.5 --class_weight --reason_flag \
  --mtl --use_clone --hidden_l 400 --hidden_a 400 --hidden_v 400 --persona_l_heads 8 --persona_a_heads 8 --persona_v_heads 8 \
  --persona_l_layer 1 --persona_a_layer 1 --persona_v_layer 1 --interactive_layer 1 --interactive_heads 4 --dropout_forward 0.3 \
  --dropout_persona_lstm_modeling 0.2 --dropout_interactive 0.2 --dropout_persona 0.2 --erc_windows 1 --shift_windows 1 \
  --persona_transform --interactive_windows 1 --epochs 140 --seed 6500
```

2. train PCGNet on MELD for ERC task
```shell
python code/run_train_erc.py --dataset MELD --data_dir ./data/meld/MELD_features_raw1.pkl \
  --valid_rate 0.0 --modals avl --lr 0.0001 --batch-size 32 --l2 0.0001 \
  --mtl --use_clone --hidden_l 200 --hidden_a 200 --hidden_v 200 --persona_transform\
  --persona_l_heads 4 --persona_a_heads 4 --persona_v_heads 4 \
  --persona_l_layer 1 --persona_a_layer 1 --persona_v_layer 1 \
  --interactive_layer 1 --interactive_heads 4 \
  --dropout_forward 0 --dropout_persona_lstm_modeling 0.2 --dropout_interactive 0.2 --dropout_persona 0.2 \
  --erc_windows 1 --shift_windows 1 --interactive_windows 1  --epochs 30 --seed 11407
```

3. evaluation PCGNet on IEMOCAP for ERC task
```shell
python code/inference.py --dataset IEMOCAP --data_dir data/iemocap/IEMOCAP_features.pkl \
  --valid_rate 0.0 --modals avl --lr 0.0001 --batch-size 32 --l2 0.0001 --dropout 0.2 --gamma 0.5 --class_weight --reason_flag \
  --mtl --use_clone --hidden_l 400 --hidden_a 400 --hidden_v 400 --persona_l_heads 8 --persona_a_heads 8 --persona_v_heads 8 \
  --persona_l_layer 1 --persona_a_layer 1 --persona_v_layer 1 --interactive_layer 1 --interactive_heads 4 --dropout_forward 0.3 \
  --dropout_persona_lstm_modeling 0.2 --dropout_interactive 0.2 --dropout_persona 0.2 --erc_windows 1 --shift_windows 1 \
  --persona_transform --interactive_windows 1 --seed 6500 --ckpt checkpoints/IEMOCAP_ckpt.pkl
```

4. evaluation PCGNet on MELD for ERC task
```shell
python code/inference.py --dataset MELD --data_dir ./data/meld/MELD_features_raw1.pkl \
  --valid_rate 0.0 --modals avl --lr 0.0001 --batch-size 32 --l2 0.0001 \
  --mtl --use_clone --hidden_l 200 --hidden_a 200 --hidden_v 200 --persona_transform\
  --persona_l_heads 4 --persona_a_heads 4 --persona_v_heads 4 \
  --persona_l_layer 1 --persona_a_layer 1 --persona_v_layer 1 \
  --interactive_layer 1 --interactive_heads 4 \
  --dropout_forward 0 --dropout_persona_lstm_modeling 0.2 --dropout_interactive 0.2 --dropout_persona 0.2 \
  --erc_windows 1 --shift_windows 1 --interactive_windows 1 --seed 11407 --ckpt checkpoints/MELD_ckpt.pkl
```


## Citation
If you find our work useful for your research, please kindly cite our paper as follows:
```
@inproceedings{tu2024persona,
title = {A Persona-Infused Cross-Task Graph Network for Multimodal Emotion Recognition with Emotion Shift Detection in Conversations},
author = {Tu, Geng and Xiong, Feng and Liang, Bin and Xu, Ruifeng},
booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {2266–2270},
year = {2024}
}
```

## Acknowledgements
Special thanks to the following authors for their contributions through open-source implementations.

* [Emotion Recognition in Conversations](https://github.com/declare-lab/conv-emotion)
* [An Open-source Benchmark of Deep Learning Models for Audio-visual Apparent and Self-reported Personality Recognition](https://github.com/liaorongfan/DeepPersonality.git)
* [Multivariate, Multi-frequency and Multimodal: Rethinking Graph Neural Networks for Emotion Recognition in Conversation](https://github.com/feiyuchen7/M3NET)
