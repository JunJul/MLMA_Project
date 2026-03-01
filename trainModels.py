from pipeline import Pipeline
from metrics import classification_result


def main(conf_dir):
    # train, validation, and test
    model = Pipeline(conf_dir)
    # model.train()
    preds, _ = model.predict()

    # get true label
    encoder = model.test_loader.dataset.encoder
    y_true = [encoder[label] for label in model.test_loader.dataset.df["class"]]

    # check performance of the model
    model_name = conf_dir.split('/')[-1].replace('.yaml', '')
    saved_path = model.experiment_dir
    classification_result(y_true, preds, model_name, saved_path)
    pass



if __name__ == "__main__":
    main("model_confs/ResNetSE.yaml")