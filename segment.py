import cv2
import numpy as np
import math
import matplotlib.pylab as plt
from numpy.core.defchararray import add
import tqdm

SIGMA = 0.8
DOWN = 0
RIGHT = 1
MAX_NUM = 70
MIN_NUM = 50
K = 150
MIN_SIZE = 50

def diff(image, v1, v2):
    return np.abs(int(image[v1]) - int(image[v2]))

def construct_segmentaion(v1, v2, cur_diff, labels, threshold, step, min_size):
    #两个点在同一个划分中
    label1 = labels[v1[0], v1[1]]
    label2 = labels[v2[0], v2[1]]
    if label1 == label2:
       return 0

    x_list, y_list = np.where(labels == label1)
    component1 = [v for v in zip(x_list, y_list)]
    x_list, y_list = np.where(labels == label2)
    component2 = [v for v in zip(x_list, y_list)]
    if step == 0:
        if cur_diff <= min(threshold[label1], threshold[label2]):
            for v in component2:
                labels[v] = label1
            threshold[label1] = cur_diff + K / (len(component1) + len(component2))
            return 1
        else:
            return 0
    else:
        if min(len(component1), len(component2)) < min_size:
            for v in component2:
                labels[v] = label1
            threshold[label1] = cur_diff + K / (len(component1) + len(component2))
            return 1
        else:
            return 0
    

def segmentation(image):
    height, width = image.shape
    edges = np.zeros((height, width, 2))
    #考虑4邻域，计算下和右的距离。
    for i in range(height):
        for j in range(width):
            if i < height - 1:
                diff_down = diff(image, (i, j), (i + 1, j))
                edges[i, j, DOWN] = diff_down
            else:
                edges[i, j, DOWN] = math.inf
            if j < width - 1:
                diff_right= diff(image, (i, j), (i, j + 1))
                edges[i, j, RIGHT] = diff_right
            else:
                edges[i, j, RIGHT] = math.inf
    #diff从小到大排序, 其中有 height + width 个 inf， 不考在排序中
    edges_rank = np.column_stack(np.unravel_index(np.argsort(edges.ravel())[: - height - width], edges.shape))
    #初始化 label
    labels = np.array(range(height * width)).reshape((height, width))
    label_num = height * width
    thredshold = np.full((height * width), K)
    #先合并到最大划分数
    for idx in tqdm.tqdm(range(len(edges_rank)), desc='First step', leave=False):
        edge = tuple(edges_rank[idx])
        if edge[2] == DOWN:
            label_num -= construct_segmentaion((edge[0], edge[1]), (edge[0] + 1, edge[1]), edges[edge], labels, thredshold, 0, MIN_SIZE)
        else:
            label_num -= construct_segmentaion((edge[0], edge[1]), (edge[0], edge[1] + 1), edges[edge], labels, thredshold, 0, MIN_SIZE)

        if label_num <= MAX_NUM:
            break
    #再合并过小的划分
    add_size = 0
    while 1:
        for idx in tqdm.tqdm(range(len(edges_rank)), desc='Second step, current label_sum = {}'.format(label_num), leave=False):
            edge = tuple(edges_rank[idx])
            if edge[2] == DOWN:
                label_num -= construct_segmentaion((edge[0], edge[1]), (edge[0] + 1, edge[1]), edges[edge], labels, thredshold, 1, MIN_SIZE + add_size)
            else:
                label_num -= construct_segmentaion((edge[0], edge[1]), (edge[0], edge[1] + 1), edges[edge], labels, thredshold, 1, MIN_SIZE + add_size)
            if label_num <= MAX_NUM:
                break

        if label_num <= MAX_NUM:
                break
        add_size += 50
    return labels

def is_front(mask, component):
    front_cnt = 0
    for c in component:
        if mask[c] == 255:
            front_cnt += 1
    return front_cnt >= len(component) / 2

def IOU(labels, mask):
    labels_set = set(labels.flatten())
    label_mask = np.full((mask.shape[0], mask.shape[1]), 0)

    for l in labels_set:
        x_list, y_list = np.where(labels == l)
        component = [v for v in zip(x_list, y_list)]

        if is_front(mask, component):
            for c in component:
                label_mask[c] = 255

    x_list, y_list = np.where(label_mask == 255)
    label_mask_front = set([v for v in zip(x_list, y_list)])
    x_list, y_list = np.where(mask == 255)
    mask_front = set([v for v in zip(x_list, y_list)])
    return len(label_mask_front & mask_front) / len(label_mask_front | mask_front)



if __name__ == "__main__":
    total_score = 0
    for i in range(12, 1000, 100):
        print('Generating segment/{}.png'.format(i))
        image = cv2.imread('data/imgs/{}.png'.format(i), 0)
        mask = cv2.imread('data/gt/{}.png'.format(i), 0)

        image = cv2.GaussianBlur(image, (5,5), SIGMA).astype(np.float64)
        labels = segmentation(image)
        score = IOU(labels, mask)
        print('IOU = ', score, '\n')
        total_score += score

        plt.matshow(labels, cmap=plt.get_cmap('gnuplot2'))
        plt.savefig('segment/{}.png'.format(i))
    print('average IOU = ', total_score / 10)