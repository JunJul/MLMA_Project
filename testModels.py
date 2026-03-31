import argparse
from pipeline import Pipeline
from metrics import classification_result

def main(conf_dir, policy):
    model = Pipeline(conf_dir, policy=policy)
    
    y_true, y_probs, test_loss = model.predict()
    
    print(f"\nTesting Completed. Average Test Loss: {test_loss:.4f}")

    model_name = str(model.experiment_dir.name)
    saved_path = model.experiment_dir
    
    classification_result(y_true, y_probs, model_name, saved_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained multi-label medical image classifier.")
    
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        default="model_confs/ResNetSE.yaml", 
        help="Path to the model configuration YAML file. (Default: model_confs/ResNetSE.yaml)"
    )
    
    parser.add_argument(
        "-p", "--policy",
        type=str,
        choices=["U-Ones", "U-Zeroes", "U-Smooth"], 
        default=None, 
        help="Override the CheXpert loss policy (U-Ones, U-Zeroes, or U-Smooth)."
    )
    
    args = parser.parse_args()
    
    main(args.config, args.policy)