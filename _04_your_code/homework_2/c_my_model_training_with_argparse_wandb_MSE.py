import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from datetime import datetime
import wandb
import argparse
import os

from pathlib import Path
BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl
print("BASE_PATH:", BASE_PATH)

import sys
sys.path.append(BASE_PATH)

from _03_homeworks.homework_2.titanic_dataset \
  import get_preprocessed_dataset


def get_data():
  train_dataset, validation_dataset, test_dataset = get_preprocessed_dataset()
  print(len(train_dataset), len(validation_dataset), len(test_dataset))

  train_data_loader = DataLoader(dataset=train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
  validation_data_loader = DataLoader(dataset=validation_dataset, batch_size=len(validation_dataset))
  test_data_loader = DataLoader(dataset= test_dataset, batch_size=len(test_dataset))
  return train_data_loader, validation_data_loader, test_data_loader


class MyModel(nn.Module):
  def __init__(self, n_input, n_output):
    super().__init__()

# MSELoss용
    self.model = nn.Sequential(
      nn.Linear(n_input, wandb.config.n_hidden_unit_list[0]),
      nn.ReLU(),
      nn.Linear(wandb.config.n_hidden_unit_list[0], wandb.config.n_hidden_unit_list[1]),
      nn.ReLU(),
      nn.Linear(wandb.config.n_hidden_unit_list[1], n_output),
    )

  def forward(self, x):
    x = self.model(x)
    return x


def get_model_and_optimizer():
  my_model = MyModel(n_input=10, n_output=1)
  optimizer = optim.SGD(my_model.parameters(), lr=wandb.config.learning_rate)

  return my_model, optimizer


def training_loop(model, optimizer, train_data_loader, validation_data_loader, test_data_loader):
  n_epochs = wandb.config.epochs
  loss_fn = nn.MSELoss()  # Use a built-in loss function
  # loss_fn = nn.BCEWithLogitsLoss()
  next_print_epoch = 100

  for epoch in range(1, n_epochs + 1):
    loss_train = 0.0
    num_trains = 0
    for train_batch in train_data_loader:
      input = train_batch['input']
      target = train_batch['target'].float().unsqueeze(1)  # Float 변환 & [512] → [512, 1]
      output_train = model(input)
      loss = loss_fn(output_train, target)
      loss_train += loss.item()
      num_trains += 1

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()


    loss_validation = 0.0
    num_validations = 0
    with torch.no_grad():
      for validation_batch in validation_data_loader:
        # input, target = validation_batch
        input = validation_batch['input']
        output_validation = model(input)
        target = validation_batch['target'].float().unsqueeze(1)
        loss = loss_fn(output_validation, target)
        loss_validation += loss.item()
        num_validations += 1

    wandb.log({
      "Epoch": epoch,
      "Training loss": loss_train / num_trains,
      "Validation loss": loss_validation / num_validations
    })

    if epoch >= next_print_epoch:
      print(
        f"Epoch {epoch}, "
        f"Training loss {loss_train / num_trains:.4f}, "
        f"Validation loss {loss_validation / num_validations:.4f}"
      )
      next_print_epoch += 100

def predict_test(model, test_data_loader):
  """Test 데이터로 예측 수행"""
  model.eval()
  predictions = []
  
  with torch.no_grad():
    for test_batch in test_data_loader:
      input = test_batch['input']
      output = model(input)
      predictions.append(output)
  
  predictions = torch.cat(predictions, dim=0).squeeze()  # [N, 1] → [N]
  return predictions


def save_predictions_to_csv(predictions, filename="submission.csv"):
  """예측 결과를 CSV로 저장 (Kaggle 제출 형식)"""
  import pandas as pd
  
  # Kaggle test.csv의 PassengerId 읽기
  CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
  test_data_path = os.path.join(CURRENT_FILE_PATH, "test.csv")
  test_df = pd.read_csv(test_data_path)
  
  # 예측값을 0 또는 1로 변환 (이진 분류)
  predictions_binary = (predictions > 0.5).long().numpy()
  
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
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'learning_rate': 1e-3,
    'n_hidden_unit_list': [20, 20],
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

  train_data_loader, validation_data_loader, test_data_loader = get_data()

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

