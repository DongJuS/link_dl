import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from datetime import datetime
import wandb
import argparse
import os
import xgboost as xgb

from pathlib import Path
BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl
print("BASE_PATH:", BASE_PATH)

import sys
sys.path.append(BASE_PATH)

from _03_homeworks.homework_2.titanic_dataset \
  import get_preprocessed_dataset


def get_data_local(batch_size=512):
  train_dataset, validation_dataset, test_dataset = get_preprocessed_dataset()
  print(len(train_dataset), len(validation_dataset), len(test_dataset))

  train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
  validation_data_loader = DataLoader(dataset=validation_dataset, batch_size=len(validation_dataset))
  test_data_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset))
  return train_data_loader, validation_data_loader, test_data_loader


def get_model_and_optimizer():
  model = xgb.XGBClassifier(
    n_estimators=wandb.config.n_estimators,
    learning_rate=wandb.config.learning_rate,
    max_depth=wandb.config.max_depth,
    random_state=42
  )
  return model, None  # optimizer 불필요

def training_loop(model, optimizer, train_data_loader, validation_data_loader, test_data_loader):
  from sklearn.metrics import log_loss
  
  X_train = torch.cat([batch['input'] for batch in train_data_loader]).numpy()
  y_train = torch.cat([batch['target'] for batch in train_data_loader]).numpy()
  
  X_val = torch.cat([batch['input'] for batch in validation_data_loader]).numpy()
  y_val = torch.cat([batch['target'] for batch in validation_data_loader]).numpy()
  
  # 매 에포크마다 학습
  for i in range(wandb.config.n_estimators):
    temp_model = xgb.XGBClassifier(
      n_estimators=i+1,
    )
    temp_model.fit(X_train, y_train)
    
    train_pred_proba = temp_model.predict_proba(X_train)[:, 1]
    val_pred_proba = temp_model.predict_proba(X_val)[:, 1]
    
    train_loss = log_loss(y_train, train_pred_proba)
    val_loss = log_loss(y_val, val_pred_proba)
    
    wandb.log({
      "Epoch": i + 1,
      "Training loss": train_loss,
      "Validation loss": val_loss
    })
    
    if (i + 1) % 10 == 0:
      print(f"Epoch {i+1}, Train: {train_loss:.4f}, Val: {val_loss:.4f}")
  
  # 마지막 모델 반환
  model.n_estimators = wandb.config.n_estimators
  model.fit(X_train, y_train)
def predict_test(model, test_data_loader):
  X_test = torch.cat([batch['input'] for batch in test_data_loader]).numpy()
  predictions = model.predict(X_test)  # 바로 0/1 반환
  return predictions


def save_predictions_to_csv(predictions, filename="submission.csv"):
  """예측 결과를 CSV로 저장 (Kaggle 제출 형식)"""
  import pandas as pd
  
  # Kaggle test.csv의 PassengerId 읽기
  CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
  test_data_path = os.path.join(CURRENT_FILE_PATH, "test.csv")
  test_df = pd.read_csv(test_data_path)
  
  # 예측값을 0 또는 1로 변환 (이진 분류)
  predictions_binary = predictions
  
  # 제출 파일 생성
  submission_df = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions_binary
  })
  
  submission_df.to_csv(filename, index=False)
  print(f"✅ Predictions saved to {filename}")
  return submission_df

def main(args):
  current_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

  config = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 3,
    'learning_rate': 0.05,
    }

  wandb.init(
    mode="online" if args.wandb else "disabled",
    project="my_model_training",
    notes="My first wandb experiment",
    tags=["my_model", "Titanic"],
    # name=current_time_str,
    name = args.name,
    config=config
  )
  print(args)
  print(wandb.config)

  train_data_loader, validation_data_loader, test_data_loader = get_data_local()

  linear_model, optimizer = get_model_and_optimizer()

  print("#" * 50, 1)

  training_loop(
    model=linear_model,
    optimizer=optimizer,
    train_data_loader=train_data_loader,
    validation_data_loader=validation_data_loader,
    test_data_loader=test_data_loader
  )
  # Test 예측 및 CSV 저장
  print("\n" + "#" * 50)
  print("Generating predictions for test data...")
  test_predictions = predict_test(linear_model, test_data_loader)
  
  # 파일명에 run name 포함
  filename = f"submission_{args.name if args.name else current_time_str}.csv"
  save_predictions_to_csv(test_predictions, filename)

  wandb.finish()


# https://docs.wandb.ai/guides/track/config
if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--wandb", action=argparse.BooleanOptionalAction, default=False, help="True or False"
  )

  parser.add_argument(
    "-b", "--batch_size", type=int, default=512, help="Batch size (int, default: 512)"
  )

  parser.add_argument(
    "-e", "--epochs", type=int, default=1_000, help="Number of training epochs (int, default:1_000)"
  )

  parser.add_argument(
    "-n", "--name", type=str, default=None, help="Run name for WandB"
  )

  args = parser.parse_args()

  main(args)

