import os
import cv2
import h5py
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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

    return colors, colors_dict


class KPVideo:
    def __init__(self, kp_file, video_file):
        # TODO
        #  make sure naming scheme of paths is well defined
        #  make it possible to provide name
        #  make sure video and pose file fit together (create warning)
        #  or print out which files are taken together?
        self.name = os.path.split(kp_file)[-1].split('.')[0]

        self.kp_dataframe = get_df_from_h5(kp_file)
        self.kp_labels = get_kp_labels(self.kp_dataframe)
        self.kp_colors, self.kp_colors_dict = get_kp_colors(self.kp_labels)

        self.video_file = video_file

    # TODO: adjust plots
    #  ability to save
    #  change fig size
    #  title according to name/ self defined titles
    #  self defined keypoint list
    #  ...

    def plot_likelihoods(self):  # TODO: adjust plot
        grid = sns.FacetGrid(self.kp_dataframe, col="keypoint", hue="keypoint", col_wrap=5, palette=self.kp_colors)
        grid.map(plt.plot, 'frame', "likelihood")
        grid.map(plt.axhline, y=0.9, color='grey', linestyle='-')
        grid.set_titles(col_template="{col_name}")
        grid.fig.tight_layout(w_pad=3)

    def plot_average_likelihoods(self):  # TODO: adjust plot
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.boxplot(x="keypoint", y="likelihood",
                    palette=self.kp_colors,
                    data=self.kp_dataframe)  # .set(title='')
        plt.xticks(rotation=90)
        ax.set(xlabel=None)
        plt.axhline(y=0.9, color='grey', linestyle='-')
        fig.tight_layout(w_pad=3)

    def frames(self):  # TODO
        pass

    def videos(self):  # TODO
        pass

    def frame_with_keypoints(self, frame, keypoints=None, save_path=''):

        # TODO
        #  fix cv2 import
        #  use correct opencv version for old osx
        #  remove and reinstall from poetry
        # https://stackoverflow.com/questions/60254766/opencv-giving-an-error-whenever-import-cv2-is-used

        if not keypoints:
            keypoints = self.kp_labels

        vidcap = cv2.VideoCapture(self.video_file)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, _frame = vidcap.read()

        if ret:
            _frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)  # correct color between plt and cv2
        else:
            pass  # TODO

        for keypoint in keypoints:
            plt.imshow(_frame, cmap='binary')
            x, y = self.kp_dataframe.query('keypoint == @keypoint and frame ==@frame')[['x', 'y']].to_numpy()[0]
            plt.plot(x, y, marker="o", markersize=5, c=self.kp_colors_dict[keypoint])
            plt.axis('off')

        if not save_path:
            plt.show()
        else:
            frame = f'{frame:010d}'  # adjust file name for correct sorting
            plt.savefig(f'{save_path}/{frame}.png', bbox_inches='tight')
            # plt.savefig(f'{save_path}/{frame}.png', bbox_inches='tight')

            plt.close()


class KPVideos:
    def __init__(self, kp_files, video_files):
        self.kpvs = list()
        # TODO if only one video is provided use it for all files
        for kp_file, video_file in zip(kp_files, video_files):
            self.kpvs.append(KPVideo(kp_file, video_file))

    def plot_average_likelihoods_across_files(self):
        model_dfs = []
        for kpv in self.kpvs:
            # print(kpv.name)
            df = kpv.kp_dataframe.copy()
            df['file'] = kpv.name
            model_dfs.append(df)

        dff = pd.concat(model_dfs, axis=0)
        g = sns.catplot(data=dff,
                        x="file", y="likelihood", col="keypoint",
                        col_wrap=3, sharey=False, palette='cubehelix',
                        kind="box", aspect=2)  # 'boxen' works, but 'violin' doesn't?
        g.set_titles("{col_name}")
        # TODO
        #  find a better way to rotate x-labels
        #  return g for saving?
        [plt.setp(ax.get_xticklabels(), rotation=90) for ax in g.axes.flat]
