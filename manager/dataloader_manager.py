import os
import dataset


def get_training_data(dataset, data_path, options_train):
    assert os.path.exists(data_path)

    if dataset == "Synapse":
        from dataset.synapse import DataLoaderTrain

    else:
        raise Exception("dataset name error!")

    return DataLoaderTrain(os.path.join(data_path, dataset), options_train)

def get_validation_data(dataset, data_path):
    assert os.path.exists(data_path)

    if dataset == "Synapse":
        from dataset.synapse import DataLoaderVal

    else:
        raise Exception("dataset name error!")

    return DataLoaderVal(os.path.join(data_path, dataset))
