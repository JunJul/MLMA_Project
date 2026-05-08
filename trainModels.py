import argparse
from pipeline import Pipeline

def main(conf_dir, policy):
    model = Pipeline(conf_dir, policy=policy)
    model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a multi-label medical image classifier.")
    
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        default="model_confs/ResNetSE.yaml",  
        help="Path to the model configuration YAML file."
    )
    
    # NEW: Add the policy override argument
    parser.add_argument(
        "-p", "--policy",
        type=str,
        choices=["U-Ones", "U-Zeros", "U-Smooth", "U-Ignore"],
        default=None,
        help="Override the CheXpert loss policy (U-Ones, U-Zeros, U-Smooth, U-Ignore)."
    )
    
    args = parser.parse_args()
    main(args.config, args.policy)