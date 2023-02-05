import h5py
import matplotlib
import numpy as np
import pandas as pd


# taken from https://stackoverflow.com/questions/51548551/reading-nested-h5-group-into-numpy-array
def traverse_h5(hdf_file):
    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset):  # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group):  # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    for path, _ in h5py_dataset_iterator(hdf_file):
        yield path


def get_df_from_h5(path):  # TODO: error handling
    keypoints_h5 = {'keypoint': [], 'likelihood': [], 'x': [], 'y': []}

    with h5py.File(path, 'r') as f:
        first = True

        for dset in traverse_h5(f):
            identifier = dset.split('/')[-1]  # identifiers: likelihood, x, y
            keypoint = dset.split('/')[-2]
            data = np.asarray(f[dset])

            if first:
                first = False
                keypoints_h5['keypoint'].append(keypoint)

            if keypoints_h5['keypoint'][-1] == keypoint:
                keypoints_h5[identifier].append(data)
            else:
                keypoints_h5['keypoint'].append(keypoint)
                keypoints_h5[identifier].append(data)

    df = pd.DataFrame(keypoints_h5).explode(['likelihood', 'x', 'y']).reset_index(drop=True)
    df['frame'] = df.groupby('keypoint').cumcount()
    return df[['frame', 'keypoint', 'likelihood', 'x', 'y']]


def get_kp_labels(dataframe):
    keypoints = set(dataframe.keypoint.values.tolist())
    keypoints = list(keypoints)
    keypoints.sort()
    return keypoints


def get_kp_colors(labels):
    num_colors = len(labels)
    cmap = matplotlib.cm.get_cmap("jet")
    colornorm = matplotlib.colors.Normalize(vmin=0, vmax=num_colors)
    colors = cmap(colornorm(np.arange(num_colors)))

    colors_dict = dict()
    for i, label in enumerate(labels):
        colors_dict[label] = colors[i]

    return colors_dict


class KPVideo:
    def __init__(self, kp_file, video_file):
        self.kp_dataframe = get_df_from_h5(kp_file)
        self.kp_labels = get_kp_labels(self.kp_dataframe)
        self.kp_colors = get_kp_colors(self.kp_labels)

        self.video_file = video_file

    def frames(self):  # TODO
        pass

    def videos(self):  # TODO
        pass
