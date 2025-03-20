#!/bin/bash
python cs336_basics/train.py --train_data ./data/tiny_stories_train_tokens.npy --val_data data/tiny_stories_valid_tokens.npy --total_steps 10000 --warmup_steps 100 --log_interval 100 --val_interval 500 --batch_size 128 --learning_rate 0.01 --save_path checkpoints/warmup_100.pt --run_name 'warmup=100'
python cs336_basics/train.py --train_data ./data/tiny_stories_train_tokens.npy --val_data data/tiny_stories_valid_tokens.npy --total_steps 10000 --warmup_steps 200 --log_interval 100 --val_interval 500 --batch_size 128 --learning_rate 0.01 --save_path checkpoints/warmup_200.pt --run_name 'warmup=200'
python cs336_basics/train.py --train_data ./data/tiny_stories_train_tokens.npy --val_data data/tiny_stories_valid_tokens.npy --total_steps 10000 --warmup_steps 400 --log_interval 100 --val_interval 500 --batch_size 128 --learning_rate 0.01 --save_path checkpoints/warmup_400.pt --run_name 'warmup=400'
python cs336_basics/train.py --train_data ./data/tiny_stories_train_tokens.npy --val_data data/tiny_stories_valid_tokens.npy --total_steps 10000 --warmup_steps 800 --log_interval 100 --val_interval 500 --batch_size 128 --learning_rate 0.01 --save_path checkpoints/warmup_800.pt --run_name 'warmup=800'
