"""Train and evaluate the RBF network using a named configuration."""

import argparse
from pathlib import Path

import yaml
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from datasets import load_dataset
from rbf_network import RBFNetwork


def read_config(path):
    with Path(path).open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def run(dataset_name, config):
    global_config = config["project"]
    dataset_config = config["datasets"][dataset_name]
    random_state = global_config["random_state"]
    X_train, X_test, y_train, y_test, classes, source = load_dataset(
        dataset_name,
        global_config["data_dir"],
        random_state,
        train_size=dataset_config.get("train_size"),
        test_size=dataset_config.get("test_size", 0.2),
    )
    print(f"Dataset: {dataset_name}\nSource: {source}\nTrain shape: {X_train.shape}\nTest shape: {X_test.shape}")

    if dataset_config.get("standardize", True):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    n_components = dataset_config.get("pca_components")
    if n_components:
        pca = PCA(n_components=n_components, random_state=random_state)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    model = RBFNetwork(random_state=random_state, **dataset_config["model"])
    model.fit(X_train, y_train)
    train_prediction, test_prediction = model.predict(X_train), model.predict(X_test)

    print("\n====================")
    print("TRAIN ACC:", accuracy_score(y_train, train_prediction))
    print("TEST ACC:", accuracy_score(y_test, test_prediction))
    print("AUTO GAMMA:", model.gamma)
    print("====================")
    print("\nClassification Report:")
    print(classification_report(y_test, test_prediction, target_names=[str(label) for label in classes]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_prediction))


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the RBF network.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--dataset", choices=["iris", "wine", "breast_cancer", "digits", "pendigits", "fashion_mnist"], default="iris")
    args = parser.parse_args()
    run(args.dataset, read_config(args.config))


if __name__ == "__main__":
    main()
