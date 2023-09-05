import os
import numpy as np
import pandas as pd

def saveascsv(identifier, _bins):
    """Base paths."""
    data_basepath = "./data/" + identifier

    """Data filepaths."""
    data_cc = [
        os.path.join(
            data_basepath, 
            "{}_cond_corr_{}_{}bins.npy".format(identifier, node_order, _bins)
        )
        for node_order in ["123", "132", "231"]
    ]
    data_cc_x = [
        os.path.join(
            data_basepath, 
            "{}_cond_corr_{}_x_{}bins.npy".format(identifier, node_order, _bins)
        )
        for node_order in ["123", "132", "231"]
    ]
    data_cc_stderr = [
        os.path.join(
            data_basepath, 
            "{}_cond_corr_{}_stderr_{}bins.npy".format(identifier, node_order, _bins)
        )
        for node_order in ["123", "132", "231"]
    ]
    data_cmi = [
        os.path.join(
            data_basepath, 
            "{}_cmi_{}_{}bins.npy".format(identifier, node_order, _bins)
        )
        for node_order in ["123", "132", "231"]
    ]
    data_cmi_x = [
        os.path.join(
            data_basepath, 
            "{}_cmi_{}_x_{}bins.npy".format(identifier, node_order, _bins)
        )
        for node_order in ["123", "132", "231"]
    ]

    csv_path_corr = [
        os.path.join(
        data_basepath, 
        "{}_cc_{}_corr_{}bins.csv".format(identifier, node_order, _bins)
        )
        for node_order in ["123", "132", "231"]
    ]

    csv_path_cmi = [
        os.path.join(
        data_basepath, 
        "{}_cmi_{}_corr_{}bins.csv".format(identifier, node_order, _bins)
        )
        for node_order in ["123", "132", "231"]
    ]

    for i in range(3):
        cc = np.load(data_cc[i])
        x = np.load(data_cc_x[i])
        stderr = np.load(data_cc_stderr[i])

        df = pd.DataFrame(
            np.vstack((x, cc, stderr)).T,
            columns=['x', 'cc', 'stderr']
        )
        df.to_csv(csv_path_corr[i], index=False)
        del df, cc, x, stderr
    
        cmi = np.load(data_cmi[i])
        x = np.load(data_cmi_x[i])

        df = pd.DataFrame(
            np.vstack((x, cmi)).T,
            columns=['x', 'cmi']
        )
        df.to_csv(csv_path_cmi[i], index=False)
        del df, cmi, x

def main():
    identifiers = [
        'motif-a',
        'motif-b',
        'motif-c',
        'motif-a_w-negative-TI',
        'motif-b_w-negative-TI',
        'motif-c_w-negative-TI',
        'motif-a_w-positive-TI',
        'motif-b_w-positive-TI',
        'motif-c_w-positive-TI'
    ]
    _bins = 25
    for id in identifiers:
        saveascsv(id, _bins)

if __name__ == "__main__":
    main()

"""End of file."""