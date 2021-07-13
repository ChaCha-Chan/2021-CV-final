import cv2
import numpy as np
import math
import matplotlib.pylab as plt
from numpy.core.defchararray import center
import tqdm
from segment import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import svm

SIGMA = 0.8

def get_RGB_hist(image):
    hist = np.zeros((8, 8, 8))
    height, width = image.shape[:2]
    for i in range(height):
        for j in range(width):
            hist[int(image[i, j, 0] / 32), int(image[i, j, 1] / 32), int(image[i, j, 2] / 32)] += 1
    hist /= height * width
    return hist.flatten()

def get_component_RGB_hist(image, component):
    hist = np.zeros((8, 8, 8))
    for c in component:
        i = c[0]
        j = c[1]
        hist[int(image[i, j, 0] / 32), int(image[i, j, 1] / 32), int(image[i, j, 2] / 32)] += 1
    hist /= len(component)
    return hist.flatten()

def get_segment_RGB_hist(image, mask):
    image = cv2.GaussianBlur(image, (5,5), SIGMA)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    labels = segmentation(gray_image)
    labels_set = set(labels.flatten())
    segment_hist = []
    segment_y = []
    for l in labels_set:
        x_list, y_list = np.where(labels == l)
        component = [v for v in zip(x_list, y_list)]
        segment_hist.append(get_component_RGB_hist(image, component))
        segment_y.append(is_front(mask, component))
    return segment_hist, segment_y

def get_hist(train_set_idx):
    total_segment_y = []
    total_merged_RGB_hist = []

    for idx in tqdm.tqdm(range(len(train_set_idx)), desc='Generating histograms', leave=False):
        i = train_set_idx[idx]
        image = cv2.imread('data/imgs/{}.png'.format(i), 1)
        mask = cv2.imread('data/gt/{}.png'.format(i), 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        global_RGB_hist = get_RGB_hist(image)
        segment_RGB_hist, segment_y = get_segment_RGB_hist(image, mask)

        merged_RGB_hist = []
        for s in segment_RGB_hist:
            merged_RGB_hist.append(np.append(global_RGB_hist, s))

        total_segment_y += segment_y
        total_merged_RGB_hist += merged_RGB_hist

    return np.array(total_merged_RGB_hist), np.array(total_segment_y)

def train_pca(total_merged_RGB_hist):
    pca = PCA(n_components=20)
    pca.fit(total_merged_RGB_hist) 
    return pca

def train_kmeans(total_pca_feature):
    clf = KMeans(n_clusters=50)
    clf.fit(total_pca_feature)
    return clf

def similarity(a, b):
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a, b)

def merge_similarity_feature(clf, total_pca_feature):
    total_segment_x = []
    centers = clf.cluster_centers_
    for i in range(total_pca_feature.shape[0]):
        similarity_feature = np.zeros(50)
        for j in range(len(centers)):
            similarity_feature[j] = similarity(total_pca_feature[i], centers[j])
        total_segment_x.append(np.append(total_pca_feature[i], similarity_feature))

    return np.array(total_segment_x)

def train_svm(x, y):
    clf = svm.SVC(kernel='linear', decision_function_shape='ovr')
    clf.fit(x, y)
    return clf

def accuracy(pred, y):
    return np.sum(pred == y) / pred.shape[0]

if __name__ == "__main__":
    #train
    train_set_idx = np.random.choice(range(1000), 200)

    total_merged_RGB_hist, total_segment_y = get_hist(train_set_idx)
    pca = train_pca(total_merged_RGB_hist)
    total_pca_feature = pca.fit_transform(total_merged_RGB_hist)

    kmeans = train_kmeans(total_pca_feature)
    total_segment_x = merge_similarity_feature(kmeans, total_pca_feature)

    svm_clf = train_svm(total_segment_x, total_segment_y)
    print('SVM score = ', svm_clf.score(total_segment_x, total_segment_y))

    #test
    test_set_idx = [i for i in range(12, 1000, 100)]
    test_hist, test_y = get_hist(test_set_idx) 
    test_pca_feature = pca.fit_transform(test_hist)
    test_x = merge_similarity_feature(kmeans, test_pca_feature)
    score = accuracy(svm_clf.predict(test_x), test_y)
    print('Accuracy = ', score)