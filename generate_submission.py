"""Functions useful to generate the submission for the challenge"""

import numpy as np
import csv


def reorder_test (ids, y_pred):
    """Based on the ids, reorder the prediction of the test set"""
    pair = dict(zip(ids, y_pred))
    pair_ordered = {k: v for k, v in sorted(pair.items(), key=lambda item: item[0])}
    pairs_np = np.array(list(pair_ordered.items()))
    ids_ord = pairs_np[:, 0]
    y_pred_ord = pairs_np[:, 1]
    return ids_ord, y_pred_ord


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def submission(ids, y_pred, name):
    """Function to call in order to generate the submissions from ids and y_pred"""
    ids = np.concatenate(ids, axis = 0)
    y_pred = np.concatenate(y_pred, axis = 0)
    ids_ord, y_pred_ord = reorder_test(ids, y_pred)
    create_csv_submission(ids_ord, y_pred_ord, name)
