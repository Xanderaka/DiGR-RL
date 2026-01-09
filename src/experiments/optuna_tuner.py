import optuna
from src.utils.config_parser import parse_config_file
from src.experiments.train_runner import run_training
import copy

def objective(trial):
    config = parse_config_file("/home2/xvreewij/code/run_configs/final-experiments/gaf_ppo_ViT_linux.yaml")
    config = copy.deepcopy(config)

    # Suggest hyperparameters
    # Optimizer + loss
    config["optim_loss"]["lr"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    config["optim_loss"]["optimizer"] = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    config["optim_loss"]["alpha"] = trial.suggest_float("alpha", 1e-6, 1e-3, log=True)
    config["optim_loss"]["beta"] = trial.suggest_float("beta", 0.0, 1.0)

    # Encoder
    config["encoder"]["latent_dim"] = trial.suggest_int("latent_dim", 10, 100, step=10)

    # Encoder (ViT)
    # config["encoder"]["activation_fn"] = trial.suggest_categorical("activation_fn", ["gelu", "relu", "leaky"])
    # config["encoder"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
    # config["encoder"]["base_channels"] = trial.suggest_categorical("base_channels", [64, 128, 256])
    # config["encoder"]["patch_size"] = trial.suggest_categorical("patch_size", [4, 8, 16])
    # config["encoder"]["vit_depth"] = trial.suggest_int("vit_depth", 4, 12)
    # config["encoder"]["vit_heads"] = trial.suggest_categorical("vit_heads", [2, 4, 8])

    #encoder (DenseNet)
    # config["encoder"]["activation_fn"] = trial.suggest_categorical("activation_fn", ["gelu", "relu", "leaky"])
    # config["encoder"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
    # config["encoder"]["initial_kernel_size"] = trial.suggest_int("initial_kernel_size", 3, 11, step=2)
    # config["encoder"]["normalization"] = trial.suggest_categorical("normalization", [True, False])
    # config["encoder"]["init_channels"] = trial.suggest_categorical("init_channels", [32, 64, 128])
    # config["encoder"]["growth_rate"] = trial.suggest_int("growth_rate", 16, 64, step=8)

    #encoder (ResNet)
    # config["encoder"]["activation_fn"] = trial.suggest_categorical("activation_fn", ["gelu", "relu", "leaky"])
    # config["encoder"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
    # config["encoder"]["initial_kernel_size"] = trial.suggest_int("initial_kernel_size", 3, 11, step=2)
    # config["encoder"]["normalization"] = trial.suggest_categorical("normalization", [True, False])
    # config["encoder"]["init_channels"] = trial.suggest_categorical("init_channels", [32, 64, 128])
    # options = {
    #     "v1": {"layers": [2, 2, 2], "channels": [64, 128, 256]},
    #     "v2": {"layers": [2, 2, 2, 2], "channels": [64, 128, 256, 512]},
    #     "v3": {"layers": [3, 4, 6], "channels": [64, 128, 256]}
    # }

    # choice_key = trial.suggest_categorical("layers_and_channels", list(options.keys()))
    # selected = options[choice_key]

    # config["encoder"]["layers_per_stage"] = selected["layers"]
    # config["encoder"]["base_channels"] = selected["channels"]


    # Domain predictor (mt)
    config["mt"]["d_predictor_activation_fn"] = trial.suggest_categorical("d_predictor_activation_fn", ["relu", "leaky", "selu", "gelu"])
    config["mt"]["d_predictor_dropout"] = trial.suggest_float("d_predictor_dropout", 0.0, 0.3)
    config["mt"]["d_predictor_num_layers"] = trial.suggest_int("d_predictor_num_layers", 1, 6)
    config["mt"]["predictor_activation_fn"] = trial.suggest_categorical("predictor_activation_fn", ["relu", "leaky", "selu", "gelu"])
    config["mt"]["predictor_dropout"] = trial.suggest_float("predictor_dropout", 0.0, 0.3)
    config["mt"]["predictor_num_layers"] = trial.suggest_int("predictor_num_layers", 1, 6)

    # Agent
    config["agent"]["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    config["agent"]["ent_coef"] = trial.suggest_float("ent_coef", 0.01, 0.1, log=True)
    config["agent"]["factor"] = trial.suggest_int("factor", 2, 4)
    config["agent"]["buffer_size"] = trial.suggest_categorical("buffer_size", [10000, 20000, 50000, 100000])
    config["agent"]["gradient_steps"] = trial.suggest_int("gradient_steps", 1, 32)
    config["agent"]["batch_size"] = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    config["agent"]["learning_starts"] = trial.suggest_int("learning_starts", 100, 5000, step=500)
    config["embed"]["start_epoch"] = trial.suggest_int("start_epoch", 10, 40)

    # Optional for quick debugging
    config["train"]["epochs"] = 50
    config["embed"]["start_epoch"] = 15
    # config["train"]["tuning"] = False

    val_loss = run_training(config)
    return val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")  # Or "maximize" for accuracy
    study.optimize(objective, n_trials=10)

    print("Best trial:")
    print(study.best_trial)