import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import yaml
import os
from dataset import load_image, ImageDataset
from utils import check_data_loader
from importlib import import_module
from tqdm import tqdm
import time
from pathlib import Path
import json

from sklearn.model_selection import train_test_split

"""
    Preprcoessing:
        normalization
        sampling method

    Modeling
        learning rate scheduler
        balance class weight

    Evaluation
        accuracy

"""

class Pipeline():
    def __init__(self, config_path, policy="U-Ones"):
        self.config = self._load_config(config_path)

        if policy is not None:
            if "params" not in self.config["loss"]:
                self.config["loss"]["params"] = {}
            self.config["loss"]["params"]["policy"] = policy

        self.experiment_dir = self._create_experiment_dir()
        self.history = {"train_loss":[], "val_loss":[]}
        self._setup()
        
    
    def _load_config(self, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def _create_experiment_dir(self):
        # model name and loss function
        model_name = self.config.get("model", {}).get("type", "unknown").split('.')[-1]
        loss_name = self.config.get("loss", {}).get("type", "unknown").split('.')[-1]
        policy_name = self.config.get("loss", {}).get("params", {}).get("policy", "unknown")

        # time when you create this folder
        timestamp = time.strftime("%Y%m%d")
        self.config["date"] = timestamp

        # experiment directory and sub directories
        exp_dir = Path(self.config.get("output_dir", "experiments")) / f"{model_name}_{loss_name}_{policy_name}"
        model_dir = exp_dir / "models"
        config_path = exp_dir / "config.yaml"
        history_path = exp_dir / "history.json"

        if exp_dir.exists() and config_path.exists():
            print(f"Resuming experiment: Found existing config at {exp_dir}")
            self._get_config_history(config_path, history_path)
        else:
            print(f"Starting new experiment: {exp_dir}")
            os.makedirs(exp_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f)
                pass

        return exp_dir

    def _get_config_history(self, config_path, history_path):
    # load existing config
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            pass
        
        # load existing hsitory
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as f:
                self.history = json.load(f)

                pass

            pass

        pass
    
    def _setup(self):
        # train on cuda if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # resume to previous epoch if this is not a new model
        self.best_epoch = self.config.get("best_epoch", 0)
        self.trained_epochs = self.config["trained_epochs"]

        # data
        data_conf = self.config["data"]
        self.train_loader, self.val_loader, self.test_loader = self._get_data_loader(data_conf)

        # model
        self.model = self._get_model()

        # convert to cuda compute
        self.model = self.model.to(self.device)

        # loss
        self.loss_fn = self._get_loss()
        
        # NEW: Ensure the loss function is also moved to the GPU!
        if isinstance(self.loss_fn, torch.nn.Module):
            self.loss_fn = self.loss_fn.to(self.device)

        # optimizer
        optim_config = self.config["optimizer"]
        self.optimizer = getattr(torch.optim, optim_config["type"])(
            self.model.parameters(),
            **optim_config["params"]
        )

        # learning rate scheduler 
        scheduler_config = self.config["scheduler"]
        self.scheduler = getattr(torch.optim.lr_scheduler, scheduler_config["type"])(
            self.optimizer, 
            **scheduler_config["params"]
        )

        # early stopping
        self.earlyStop = self._get_EarlyStop()
        
        # if there is a previosuly trained model
        model_name = self.config["model"]["type"].split('.')[-1]
        policy = self.config.get("loss", {}).get("params", {}).get("policy", "unknown")
        trained_model_path = self.experiment_dir / "models" / f"{model_name}_{policy}_epoch_{self.best_epoch}.pt"
        if trained_model_path.exists():
            print(f"Loading from {trained_model_path}")
            self.load_model(trained_model_path)
            pass

        pass


    def _get_model(self):
        model_config = self.config["model"]
        _, model_name = model_config["type"].rsplit('.', 1)

        module = import_module(model_config["type"])
        model = getattr(module, model_name)
            
        return model(**model_config.get("params", dict()))
    
    def _get_EarlyStop(self):
        stop_config = self.config["earlyStop"]
        module_path, stop_name = stop_config["type"].rsplit('.', 1)
        module = import_module(module_path, stop_name)
        earlyStop = getattr(module, stop_name)
        return earlyStop(**stop_config.get("params", dict()))


    def _get_loss(self):
        loss_config = self.config["loss"]
        module_path, fn_name = loss_config["type"].rsplit('.', 1)
        module = import_module(module_path, fn_name)
        loss_fn = getattr(module, fn_name)
        return loss_fn(**loss_config.get("params", dict()))
    

    def _criterion(self, outputs, labels):
        return self.loss_fn(outputs, labels)

    
    def _get_data_loader(self, data_conf):
        # data path
        data_dir = data_conf["data_dir"]
        train_path = data_conf["train_file"]
        val_path = data_conf["val_file"]
        test_path = data_conf["test_file"]

        def _resolve_csv_path(base_dir, fname):
            # Build candidate path
            candidate = os.path.join(base_dir, fname)

            # If it's a directory, try common csv names inside
            if os.path.isdir(candidate):
                for try_name in ("train.csv", "valid.csv", "test.csv", "train", "valid", "test"):
                    p = os.path.join(candidate, try_name)
                    if os.path.isfile(p):
                        return p

            # If candidate is a file already, return it
            if os.path.isfile(candidate):
                return candidate

            # Try adding .csv extension
            if not candidate.lower().endswith('.csv'):
                candidate_csv = candidate + '.csv'
                if os.path.isfile(candidate_csv):
                    return candidate_csv

            # Try stripping smart quotes and whitespace
            stripped = fname.strip('\u201C\u201D\"\'\n\r ')
            candidate2 = os.path.join(base_dir, stripped)
            if os.path.isfile(candidate2):
                return candidate2
            if not candidate2.lower().endswith('.csv'):
                candidate2_csv = candidate2 + '.csv'
                if os.path.isfile(candidate2_csv):
                    return candidate2_csv

            # Fallback to original join (may raise clearer error downstream)
            return os.path.join(base_dir, fname)

        train_csv = _resolve_csv_path(data_dir, train_path)
        train_df = load_image(train_csv)
        if val_path != "None":
            val_csv = _resolve_csv_path(data_dir, val_path)
            val_df = load_image(val_csv)
        else:
            train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df.get("class", None))
        
        test_csv = _resolve_csv_path(data_dir, test_path)
        test_df = load_image(test_csv)

        train_dt, val_dt, test_dt = ImageDataset(train_df), ImageDataset(val_df), ImageDataset(test_df)

        train_loader = DataLoader(train_dt, batch_size=data_conf["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dt, batch_size=data_conf["batch_size"])
        test_loader = DataLoader(test_dt, batch_size=data_conf["batch_size"])

        return train_loader, val_loader, test_loader
    

    def train(self):
        print("------ Start Training ------")
        best_val_loss = float("inf")
        _, model_name = self.config["model"]["type"].rsplit(".", 1)
        print(f"Training {model_name}, with parameters:")
        print(self.config["model"].get("params", None))

        max_epochs = self.config.get("max_epochs", 50)

        # train the modelf
        while self.trained_epochs < max_epochs:
            # clear the cahce
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self._save_history()

            log_msg = f"Epoch {self.trained_epochs}: Train loss: {train_loss:.3f}"
            log_msg += f" | Val loss: {val_loss:.3f}"
            print(log_msg)

            
            # learning rate scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # save model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_epoch = self.trained_epochs
                policy = self.config['loss']['params']['policy']
                self._save_model(self.experiment_dir / "models" / f"{model_name}_{policy}_epoch_{self.best_epoch}.pt")
                self.config["best_epoch"] = self.best_epoch

            self.trained_epochs += 1
            self.config["trained_epochs"] = self.trained_epochs
            self._save_yaml()
            
            # early stop
            self.earlyStop(val_loss)
            if self.earlyStop.early_stop:
                print(f"Early stopping triggered at epoch {self.trained_epochs}!")
                break

            pass

        return self.model, self.history
    
    def _save_history(self):
        with open(self.experiment_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4)
            pass
        pass

    def _save_yaml(self):
        with open(self.experiment_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(self.config, f)
            pass
        pass

    def _save_model(self, path):
        checkpoint = {
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'scheduler_state_dict': self.scheduler.state_dict(),
        'epoch': self.trained_epochs
        }
        torch.save(checkpoint, path)
        pass

    def load_model(self, path, model=None):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if model is not None:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        pass
    

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        train_len = len(self.train_loader)
        for _, data_point in tqdm(enumerate(self.train_loader), total=train_len):
             # data
             img, _,  y = data_point
             img = img.to(self.device)
             y = y.to(self.device)

            # forward
             logits = self.model(img)

             self.optimizer.zero_grad()

             # loss
             loss = self._criterion(logits, y)

             # backward
             loss.backward()
             self.optimizer.step()
             running_loss += loss.item()

             pass

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss


    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0

        val_len = len(self.val_loader)
        with torch.no_grad():
            for _, data_point in tqdm(enumerate(self.val_loader), total=val_len):
                # data
                img, _,  y = data_point
                img = img.to(self.device)
                y = y.to(self.device)
                
                # forward
                logits = self.model(img)

                # loss
                loss = self._criterion(logits, y)
                running_loss += loss.item()

                pass

        avg_loss = running_loss / len(self.val_loader)
        return avg_loss
    

    def predict(self):
        print("------ Start Testing ------")
        self.model.eval()
        running_loss = 0.0

        test_len = len(self.test_loader)
        
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for _, data_point in tqdm(enumerate(self.test_loader), total=test_len):
                # data
                img, _, y = data_point
                img = img.to(self.device)
                y = y.to(self.device)
                
                # forward
                logits = self.model(img)
                
                # Multi-label probabilities
                probs = torch.sigmoid(logits)
                
                # Store data on CPU
                all_probs.append(probs.cpu())
                all_targets.append(y.cpu())

                # loss
                loss = self._criterion(logits, y)
                running_loss += loss.item()

                pass

        avg_loss = running_loss / len(self.test_loader)
        
        all_probs_tensor = torch.cat(all_probs, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)
        
        return all_targets_tensor.numpy(), all_probs_tensor.numpy(), avg_loss
