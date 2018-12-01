# analyze encoding 128-dim space
# build Centroid, MaxRadius for each label
# build (label*label) tables: Distance between Centroids (DC); DC-(RmaxI+RmaxJ)

import pickle

import matplotlib.pyplot as plt
import numpy as np
from cam_detect_cfg import cfg
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder


class DimSpaceAnalyzer:

    def __init__(self):
        self.data_folder = 'face_dataset/'  # folder with label-folders and face_encodings.pkl to analyzer
        self.label_list = cfg['face_labels_list']
        # load encodings
        self.known = pickle.loads(open(self.data_folder + 'face_encodings.pkl', "rb").read())
        self.known_labels = np.array(self.known['labels'])
        self.known_encodings = np.array(self.known['encodings'])
        self.known_img_fnames = np.array(self.known['img_fnames'])

        self.encoder = LabelEncoder()
        self.encoder.fit(self.known_labels)


    def build_centroids(self):
        self.centroids = {}
        for label in self.label_list:
            label_idx = self.known_labels==label
            label_encodings = self.known_encodings[label_idx]
            kmeans = KMeans(n_clusters=1).fit(label_encodings)
            self.centroids[label]= kmeans.cluster_centers_

    def build_radiuses(self):
        self.radiuses = {}
        self.closest_dist = {}
        self.closest_fname = {}
        for label in self.label_list:
            label_idx = self.known_labels==label
            label_encodings = self.known_encodings[label_idx]
            distances = np.linalg.norm(label_encodings - self.centroids[label], axis=1)
            self.radiuses[label]     = max( distances )
            self.closest_dist[label] = min( distances )

    @staticmethod
    def distance(emb1, emb2):
        return np.sqrt(np.sum(np.square(emb1 - emb2)))

    def print_clusters_info(self):
        print('\nClusters info:')
        print(f"{'label':10} {'Rmin'}    {'Rmax'} ")
        for l in self.label_list:
            print(f'{l:10} {self.closest_dist[l]:>4.2f}    {self.radiuses[l]:>4.2f}')


    def build_centroids_distances(self):

        #y_train = self.encoder.transform(targets)
        #labels = self.encoder.inverse_transform(predicts)

        self.centr_dist = {}
        for i in self.label_list:
            self.centr_dist[i] = {}
            for j in self.label_list:
                self.centr_dist[i][j]=DimSpaceAnalyzer.distance(self.centroids[i],self.centroids[j])


    def build_clusters_gaps(self):
        pass


    def print_inter_clusters_info(self):
        print('\ninter-clusters distances:')
        print(f"{' ':10s} ",end='')
        for j in self.label_list:
            print(f"{j:>10s} ",end='')
        print(' ')

        for i in self.label_list:
            print(f'{i:10} ', end='')
            for j in self.label_list:
                print(f'{self.centr_dist[i][j]:>10.2f} ',end='')
            print(' ')

        print('------------')

        print(f'{"MIN: ":>10s} ',end='')
        for j in self.label_list:
            min_dist = min( [self.centr_dist[lbl][j] for lbl in self.label_list if self.centr_dist[lbl][j] !=0 ] )
            print(f'{min_dist:>10.2f} ', end='')
        print(' ')

        print(f'{"MAX: ":>10s} ',end='')
        for j in self.label_list:
            max_dist = max( [self.centr_dist[lbl][j] for lbl in self.label_list ])
            print(f'{max_dist:>10.2f} ', end='')
        print(' ')

    def plot_clusters(self):
        X_embedded = TSNE(n_components=2,init='pca').fit_transform(self.known_encodings)

        colors = {'Ded':'red','Igor':'blue','Olka':'green','Yulka':'purple','Yehor':'cyan'}
        for i, t in enumerate(set(self.known_labels)):
            idx = self.known_labels == t
            color = colors[t] if t in colors else 'magenta'
            plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t, c=color )

        plt.legend(bbox_to_anchor=(1, 1))
        plt.savefig('dataset_visual.png')
        plt.show()


def main():

    print(f"FYI:  distance(0*,1*) in 128-dim space = {DimSpaceAnalyzer.distance(np.zeros((1,128)),np.ones ((1,128))):.2f}")

    dsa = DimSpaceAnalyzer()

    dsa.build_centroids()
    dsa.build_radiuses()
    dsa.print_clusters_info()

    dsa.build_centroids_distances()
    dsa.build_clusters_gaps()
    dsa.print_inter_clusters_info()

    dsa.plot_clusters()

if __name__ == '__main__':
    main()