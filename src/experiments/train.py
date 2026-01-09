"""
Train module for PARS-DARLING.
Contains the Training class that handles backbone and RL-based domain embedding training,
plus evaluation, checkpointing, and adaptive RL agent switching.
"""

import copy
import math
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, Precision, F1Score, Recall, ConfusionMatrix
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

import wandb
from wandb.wandb_run import Run

from src.models.multi_task import run_heads, MultiTaskHead
from src.models.null_agent import NullAgent
from src.loss.multi_joint_loss import MultiJointLoss
from src.rl.rl_builder import build_rl
from src.ui.base_ui import BaseUI


class EPRewardLogger(BaseCallback):
    """Logs the mean episode reward for the most recent batch of RL steps."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_rew_mean = None  # stores most recent mean reward

    def _on_step(self) -> bool:
        ep_infos = self.model.ep_info_buffer  # stable-baselines internal buffer

        if len(ep_infos) > 0:
            ep_rewards = [ep_info['r'] for ep_info in ep_infos if 'r' in ep_info]
            self.ep_rew_mean = np.mean(ep_rewards) if ep_rewards else float('nan')

        return True


class Training:
    """
    Handles the full DARLInG training pipeline:
      - Backbone encoder + null head training
      - RL agent training for domain embedding
      - Validation & checkpointing
      - Adaptive PPO → SAC switching
    """

    def __init__(self,
                 encoder: nn.Module,
                 null_head: MultiTaskHead,
                 embedding_agent: str,
                 null_agent: NullAgent,
                 encoder_optimizer: Optimizer,
                 null_head_optimizer: Optimizer,
                 loss_func: MultiJointLoss,
                 logging: Run,
                 checkpoint_dir: Path,
                 ui: BaseUI,
                 embed_head_optim_class: callable,
                 lr: float,
                 reward_function: callable,
                 policy_name: str,
                 config: dict[str, dict[str, any]],
                 num_classes: int = 6,
                 agent_start_epoch: int = 0):
        """
        Args:
            encoder: CNN encoder for imaged time-series input.
            null_head: MT head for the null domain branch.
            embedding_agent: RL agent or "known" agent for domain embedding.
            null_agent: Agent producing null domain embedding.
            encoder_optimizer: Optimizer for encoder.
            null_head_optimizer: Optimizer for null head.
            loss_func: Loss function for multi-task ELBO + classification.
            logging: Weights & Biases logger.
            checkpoint_dir: Where to save model checkpoints.
            ui: UI object for progress updates.
            embed_head_optim_class: Class for embedding head optimizer.
            lr: Learning rate for optimizers.
            reward_function: RL reward function.
            policy_name: Name of RL policy architecture.
            config: Configuration dictionary.
            num_classes: Number of gesture classes.
            agent_start_epoch: Epoch index to begin RL agent training.
        """
        # Core modules
        self.encoder = encoder
        self.null_head = null_head
        self.embed_head = None  
        self.embedding_agent: str | BaseAlgorithm = embedding_agent
        self.null_agent = null_agent
        self.encoder_optimizer = encoder_optimizer
        self.null_head_optimizer = null_head_optimizer
        self.embed_head_optimizer = None
        self.embed_head_optim_class = embed_head_optim_class
        self.loss_func = loss_func

        # Logging & config
        self.logging = logging
        self.checkpoint_dir = checkpoint_dir
        self.ui = ui
        self.lr = lr
        self.reward_function = reward_function
        self.config = config
        self.num_classes = num_classes
        self.agent_start_epoch = agent_start_epoch
        self.policy_name = policy_name

        # Environment placeholders
        self.env = None
        self.valid_env = None

        # Checkpointing state
        self.best_joint_loss = 1.0e8
        self.f1_score = 0
        self.best_f1_score = 0
        self.prev_checkpoint_fp: Path | None = None
        self.prev_checkpoint_fp_f1: Path | None = None

        # Metrics
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.prec = Precision(task="multiclass", num_classes=num_classes)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.recall = Recall(task="multiclass", num_classes=num_classes)
        self.conf_mat = ConfusionMatrix(task="multiclass", num_classes=num_classes)

        # Step counter for logging
        self.step = -1

        self.embed_agent_type = "ppo" 
        self.sequential = config["agent"]["sequential"]

    def _forward_pass(self,
                      amp: torch.Tensor | None,
                      phase: torch.Tensor | None,
                      bvp: torch.Tensor,
                      gesture: torch.Tensor,
                      info: dict[str, any],
                      device: torch.device,
                      no_grad_backbone: bool,
                      domain_label: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Runs one forward pass of the model (encoder → embeddings → heads).

        Args:
            amp, phase, bvp: CSI amplitude/phase and ground-truth BVP.
            gesture: Gesture labels.
            info: Batch metadata dict.
            device: Device to run on.
            no_grad_backbone: Whether to freeze encoder gradients.
            domain_label: Domain ID labels for domain classification.

        Returns:
            Dictionary containing predictions, embeddings, and loss components.
        """
        gesture = gesture.to(device)
        amp, phase, bvp = amp.to(device), phase.to(device), bvp.to(device)

        # Encode inputs → latent z
        with torch.no_grad() if no_grad_backbone else torch.enable_grad():
            z = self.encoder(amp, phase, bvp)

        # Null embedding from the null agent
        null_embedding = self.null_agent(z, info)

        # Domain embedding (RL agent or known agent)
        if self.embed_head is not None:
            if isinstance(self.embedding_agent, BaseAlgorithm):  # RL agent
                obs = z.detach().cpu()
                action, _ = self.embedding_agent.predict(obs)
                domain_embedding = F.softmax(torch.tensor(action, device=device, dtype=torch.float32), dim=1)
            else:
                raise ValueError("Invalid embedding agent type.")

            out_dict = {"agent_action": action}
        else:
            domain_embedding = None
            out_dict = {}

        with torch.no_grad() if no_grad_backbone else torch.enable_grad():
            bvp_null, gesture_null, bvp_embed, gesture_embed = run_heads(
                self.null_head, self.embed_head, null_embedding, domain_embedding, z
            )

        loss_dict = self.loss_func(
            target_label=gesture,
            null_domain_pred=bvp_null,
            null_class_pred=gesture_null,
            embed_domain_pred=bvp_embed,
            embed_class_pred=gesture_embed,
            domain_label=domain_label
        )

        out_dict.update({
            "z": z,
            "gesture_null": gesture_null,
            "gesture_embed": gesture_embed,
            "domain_null": bvp_null,
            "domain_embed": bvp_embed,
            "class_loss": loss_dict["class_loss"],
            "domain_loss": loss_dict["domain_loss"],
            "null_loss": loss_dict["null_joint_loss"],
            "embed_loss": loss_dict["embed_joint_loss"],
            "joint_loss": loss_dict["joint_loss"]
        })
        return out_dict

    def _train_backbone(self, train_loader: DataLoader, device: torch.device, epoch: int) -> bool:
        """Trains just the backbone component of the model.

        Returns:
            True if the model training was successful, False if loss goes to nan.
        """
        self.ui.update_status("Training model...")
        for batch_idx, (amp, phase, bvp, info) in enumerate(train_loader):
            self.step += 1
            domain_label = info["domain_label"]
            if not isinstance(domain_label, torch.Tensor):
                domain_label = torch.tensor(domain_label, dtype=torch.long, device=device)
            else:
                domain_label = domain_label.to(device)
            pass_result = self._forward_pass(
                amp, phase, bvp, info["gesture"], info,
                device, no_grad_backbone=False, domain_label=domain_label
            )
            
            # Backward pass
            self.encoder_optimizer.zero_grad()
            self.null_head_optimizer.zero_grad()
            if self.embed_head_optimizer is not None:
                self.embed_head_optimizer.zero_grad()

            pass_result["joint_loss"].backward()

            self.encoder_optimizer.step()
            self.null_head_optimizer.step()
            if self.embed_head_optimizer is not None:
                self.embed_head_optimizer.step()

            # Metric calculations
            domain_loss_value = pass_result["domain_loss"].item()
            null_loss_value = pass_result["null_loss"].item()
            joint_loss_value = pass_result["joint_loss"].item()

            should_exit = (
                np.isnan(domain_loss_value)
                or np.isnan(null_loss_value)
                or np.isnan(joint_loss_value)
            )

            if pass_result["embed_loss"] is not None:
                embed_loss_value = pass_result["embed_loss"].item()
                loss_diff = embed_loss_value - null_loss_value
                if np.isnan(embed_loss_value):
                    should_exit = True
            else:
                embed_loss_value = float("nan")
                loss_diff = float("nan")

            # Prepare log values
            loss_vals = {
                "train_loss": joint_loss_value,
                "train_domain_loss": domain_loss_value,
                "train_null_loss": null_loss_value,
                "train_embed_loss": embed_loss_value,
                "train_loss_diff": loss_diff,
            }

            log_dict = {}
            log_dict.update(loss_vals)

            ui_data = {"epoch": epoch, "batch": batch_idx}
            ui_data.update(loss_vals)

            self.ui.update_data(ui_data)
            # self.logging.log(log_dict, self.step)
            self.ui.step(len(info["user"]))

            if should_exit:
                self.ui.update_status("Joint loss is nan, exiting...")
                return False

        return True

    def _validate_holistic(self, valid_loader: DataLoader, device, epoch: int) -> float:
        """Performs validation on the entire model.

        Returns:
            The average joint loss over the validation run.
        """
        self.ui.update_status("Running validation...")
        joint_losses = []
        domain_losses = []
        bvp_null_losses = []
        bvp_embed_losses = []
        gesture_gts, domain_gts = [], []
        gesture_null_preds, domain_null_preds = [], []
        gesture_embed_preds, domain_embed_preds = [], []

        for batch_idx, (amp, phase, bvp, info) in enumerate(valid_loader):
            domain_label = info["domain_label"]

            if not isinstance(domain_label, torch.Tensor):
                domain_label = torch.tensor(domain_label, dtype=torch.long, device=device)
            else:
                domain_label = domain_label.to(device)

            pass_result = self._forward_pass(amp, phase, bvp, info["gesture"], info, device, True, domain_label)

            domain_loss_value = pass_result["domain_loss"].item()
            null_loss_value = pass_result["null_loss"].item()
            joint_loss_value = domain_loss_value + null_loss_value

            if pass_result["embed_loss"] is not None:
                embed_loss_value = pass_result["embed_loss"].item()
                joint_loss_value += embed_loss_value
            else:
                embed_loss_value = None

            domain_losses.append(domain_loss_value)
            bvp_null_losses.append(null_loss_value)
            bvp_embed_losses.append(embed_loss_value) 
            joint_losses.append(joint_loss_value)

            gesture_gts.append(info["gesture"].detach())
            domain_gts.append(domain_label)

            gesture_null_preds.append(pass_result["gesture_null"])
            domain_null_preds.append(pass_result["domain_null"])

            gesture_embed_preds.append(pass_result["gesture_embed"])
            domain_embed_preds.append(pass_result["domain_embed"])

            data_dict = {
                "valid_loss": joint_loss_value,
                "loss_diff": 0.,
                "epoch": epoch,
                "batch": batch_idx
            }
            if embed_loss_value is not None:
                data_dict["embed_loss"] = embed_loss_value - null_loss_value

            self.ui.update_data(data_dict)
            self.ui.step(len(info["user"]))

        # Stack and move to CPU
        gesture_gts = torch.cat(gesture_gts).cpu()
        domain_gts = torch.cat(domain_gts).cpu()

        gesture_null_preds = torch.argmax(torch.cat(gesture_null_preds), dim=1).cpu()
        domain_null_preds = torch.argmax(torch.cat(domain_null_preds), dim=1).cpu()

        if gesture_embed_preds[0] is not None:
            gesture_embed_preds = torch.argmax(torch.cat(gesture_embed_preds), dim=1).cpu()
            domain_embed_preds = torch.argmax(torch.cat(domain_embed_preds), dim=1).cpu()
        else:
            gesture_embed_preds, domain_embed_preds = None, None

        joint_losses = np.mean(joint_losses).item()
        num_classes = len(domain_label)
        self.acc_domain = Accuracy(task="multiclass", num_classes=num_classes)
        self.prec_domain = Precision(task="multiclass", num_classes=num_classes)
        self.f1_domain = F1Score(task="multiclass", num_classes=num_classes)
        self.recall_domain = Recall(task="multiclass", num_classes=num_classes)
        log_dict = {
            "valid_domain_loss": np.mean(np.array(domain_losses)),
            "valid_joint_loss": joint_losses,
            "valid_bvp_null_loss": np.mean(np.array(bvp_null_losses)),
            "valid_bvp_embed_loss": float("nan"),
            "valid_null_gesture_acc": self.acc(gesture_gts, gesture_null_preds),
            "valid_null_gesture_f1": self.f1(gesture_gts, gesture_null_preds),
            "valid_null_gesture_prec": self.prec(gesture_gts, gesture_null_preds),
            "valid_null_gesture_recall": self.recall(gesture_gts, gesture_null_preds),
            "valid_null_domain_acc": self.acc_domain(domain_gts, domain_null_preds),
            "valid_null_domain_f1": self.f1_domain(domain_gts, domain_null_preds),
            "valid_null_domain_prec": self.prec_domain(domain_gts, domain_null_preds),
            "valid_null_domain_recall": self.recall_domain(domain_gts, domain_null_preds),
            "valid_gesture_conf_mat": self._conf_matrix(gesture_gts, gesture_null_preds),
            "valid_domain_conf_mat": self._conf_matrix(domain_gts, domain_null_preds),
            "valid_embed_gesture_acc": float("nan"),
            "valid_embed_gesture_f1": float("nan"),
            "valid_embed_gesture_prec": float("nan"),
            "valid_embed_gesture_recall": float("nan"),
            "valid_embed_domain_acc": float("nan"),
            "valid_embed_domain_f1": float("nan"),
            "valid_embed_domain_prec": float("nan"),
            "valid_embed_domain_recall": float("nan"),
        }
        f1_score = 0
        if gesture_embed_preds is not None:
            log_dict.update({
                "valid_bvp_embed_loss": np.mean(np.array(bvp_embed_losses)),
                "valid_embed_gesture_acc": self.acc(gesture_gts, gesture_embed_preds),
                "valid_embed_gesture_f1": self.f1(gesture_gts, gesture_embed_preds),
                "valid_embed_gesture_prec": self.prec(gesture_gts, gesture_embed_preds),
                "valid_embed_gesture_recall": self.recall(gesture_gts, gesture_embed_preds),
                "valid_embed_domain_acc": self.acc_domain(domain_gts, domain_embed_preds),
                "valid_embed_domain_f1": self.f1_domain(domain_gts, domain_embed_preds),
                "valid_embed_domain_prec": self.prec_domain(domain_gts, domain_embed_preds),
                "valid_embed_domain_recall": self.recall_domain(domain_gts, domain_embed_preds),
            })
            f1_score = self.f1(gesture_gts, gesture_embed_preds)

        # Cleanup
        del gesture_gts, domain_gts, gesture_null_preds, domain_null_preds
        del gesture_embed_preds, domain_embed_preds

        # plt.close("all")
        return joint_losses, f1_score

    def _conf_matrix(self, y_true, y_pred):
        """Generates a confusion matrix plot with dynamic class count."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        import numpy as np

        # Move tensors to CPU and convert to numpy arrays
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        # Automatically detect number of classes
        num_classes = max(y_true.max(), y_pred.max()) + 1
        labels = np.arange(num_classes)

        # Compute confusion matrix
        conf_mat = confusion_matrix(y_true, y_pred, labels=labels)

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")

        return fig

    def _save_checkpoint(self, curr_joint_loss: float, f1_score: float, epoch: int):
        """Saves a checkpoint if validation shows performance improvement."""
        if epoch < self.agent_start_epoch:
            return
        if self.embed_agent_type != "sac" and self.config["agent"]["sequential"]:
            return
        else:
            if curr_joint_loss < self.best_joint_loss:
                checkpoint_fp = (self.checkpoint_dir
                                / f"{self.logging.name}-ep-{epoch}-{self.embed_agent_type}.pth")
                agent_fp = (self.checkpoint_dir
                            / f"{self.logging.name}_agent-ep-{epoch}-{self.embed_agent_type}.zip")
                if self.prev_checkpoint_fp is not None:
                    if self.prev_checkpoint_fp.exists():
                        self.prev_checkpoint_fp.unlink()
                save_dict = {
                    "encoder_state_dict": self.encoder.state_dict(),
                    "null_mt_head_state_dict": self.null_head.state_dict(),
                }
                if self.embed_head is not None:
                    save_dict["embed_mt_head_state_dict"] = \
                        self.embed_head.state_dict()

                torch.save(save_dict, checkpoint_fp)
                self.embedding_agent.save(agent_fp)
                self.best_joint_loss = curr_joint_loss
                self.prev_checkpoint_fp = checkpoint_fp
            if f1_score > self.best_f1_score:
                checkpoint_fp = (self.checkpoint_dir
                                / f"{self.logging.name}-f1-ep-{epoch}-{self.embed_agent_type}.pth")
                agent_fp = (self.checkpoint_dir
                            / f"{self.logging.name}_agent-f1-ep-{epoch}-{self.embed_agent_type}.zip")
                if self.prev_checkpoint_fp_f1 is not None:
                    if self.prev_checkpoint_fp_f1.exists():
                        self.prev_checkpoint_fp_f1.unlink()
                save_dict = {
                    "encoder_state_dict": self.encoder.state_dict(),
                    "null_mt_head_state_dict": self.null_head.state_dict(),
                }
                if self.embed_head is not None:
                    save_dict["embed_mt_head_state_dict"] = \
                        self.embed_head.state_dict()

                torch.save(save_dict, checkpoint_fp)
                self.embedding_agent.save(agent_fp)
                self.best_f1_score = f1_score
                self.prev_checkpoint_fp_f1 = checkpoint_fp
            

    def train(self,
              train_loader: DataLoader,
              valid_loader: DataLoader,
              epochs: int,
              agent_epochs: int,
              device: torch.device):
        """Trains the model which was given in the initialization of the class.

        Args:
            train_loader: DataLoader containing the training data.
            valid_loader: DataLoader containing the validation data.
            epochs: Number of epochs to train for.
            agent_epochs: Number of epochs to train the embedding agent for.
            device: Which device to train on.

        Returns:
            best loss if training succeeded or else 0
        """
        # Ensure that the models are on the right devices
        self.encoder.to(device)
        self.null_head.to(device)

        train_embedding_agent = self.embedding_agent != "known"
        def extract_all_domains_from_loader(loader):
            """Extract all unique domain combinations from a dataloader."""
            domain_set = set()
            
            for _, _, _, info in loader:
                # Assume info contains lists or batched elements
                for user, torso, face, room in zip(
                    info["user"],
                    info["torso_location"],
                    info["face_orientation"],
                    info["room_num"]
                ):
                    domain_tuple = (user, torso, face, room)
                    domain_set.add(domain_tuple)
            
            return domain_set
        total_agent_timesteps = 0
        train_domains = extract_all_domains_from_loader(train_loader)
        val_domains = extract_all_domains_from_loader(valid_loader)
        all_domains = sorted(set(list(train_domains) + list(val_domains)))  # list of all user IDs used

        self.domainid = {domain: idx for idx, domain in enumerate(all_domains)}
        recent_rewards = []
        reward_stability_window = self.config["agent"]["window"]
        threshold = self.config["agent"]["threshold"]
        self.embed_agent_type = "ppo"
        self.switch_to_sac = False
        self.switch_epoch = self.config["agent"]["switch_epoch"]
        for epoch in range(epochs):
            if train_embedding_agent and epoch >= self.agent_start_epoch:
                # Check if we should build the agent or switch to SAC agent
                if epoch == self.agent_start_epoch or self.switch_to_sac:
                    # Duplicate the null head to create the embedding head
                    self.ui.update_status("Training embedding agent starting "
                                          "this epoch...")
                    self.embed_head = copy.deepcopy(self.null_head)
                    self.embed_head_optimizer = self.embed_head_optim_class(
                        self.embed_head.parameters(),
                        lr=self.lr
                    )
                    self.embed_head.to(device)
                    rl = build_rl(self.encoder, self.null_head, self.embed_head,
                                  self.null_agent, self.embedding_agent, device,
                                  train_loader, self.reward_function, agent_epochs, 
                                  self.embed_agent_type, self.switch_to_sac, self.policy_name, self.config)
                    self.env, self.embedding_agent, total_agent_timesteps, self.embed_agent_type, self.switch_to_sac = rl
                # Put backbone encoder to eval mode
                self.encoder.eval()
                self.null_head.eval()
                if self.embed_head is not None:
                    self.embed_head.eval()

                self.ui.update_status(f"Training agent for "
                                      f"{total_agent_timesteps} steps...")

                callback = EPRewardLogger(verbose=1)
                self.embedding_agent.learn(total_timesteps=total_agent_timesteps, callback=callback)
                ep_rew_mean = callback.ep_rew_mean
                if self.sequential:
                    if self.embed_agent_type == "ppo":
                        recent_rewards.append(ep_rew_mean)
                        if len(recent_rewards) >= self.switch_epoch:
                            reward_std = np.std(recent_rewards[-reward_stability_window:])
                            print(f"Reward std over last {reward_stability_window} epochs: {reward_std:.2f}")
                            if reward_std < threshold:
                                print("Reward stable. Switching to SAC next epoch.")
                                self.switch_to_sac = True

            # Train the backbone portion of the model
            self.encoder.train()
            self.null_head.train()
            if self.embed_head is not None:
                self.embed_head.train()

            if not self._train_backbone(train_loader, device, epoch):
                return 0

            # Put backbone to eval mode
            self.encoder.eval()
            self.null_head.eval()
            if self.embed_head is not None:
                self.embed_head.eval()

            # Perform validation
            curr_joint_loss, f1_score = self._validate_holistic(
                valid_loader, device, epoch
            )
            if math.isnan(curr_joint_loss):
                self.ui.update_status("Validation loss is NaN.")
                return 0

            # Save checkpoint
            self._save_checkpoint(curr_joint_loss, f1_score, epoch)
        return self.best_joint_loss
