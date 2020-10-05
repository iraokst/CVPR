import numpy as np
import cv2 as cv
import os
import time
import pandas as pd
from shapely.geometry import Polygon, Point
from scipy.spatial import distance

working_directory = os.getcwd()

def experiment(descriptor=cv.AKAZE_create(),
               matcher=cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_SL2),
               source_dir='dataset3',
               ideal_name='ideal.jpg', result_dir='Results',
               dist_param=0.74, ransac_param=15e-4, start_numerate=1):
    try:
        os.mkdir(working_directory + '\\' + result_dir)
    except FileExistsError:
        pass

    title = str(type(descriptor)).split()[1].strip('\'>')
    ideal = cv.imread(ideal_name)
    h, w, d = ideal.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    gray_ideal = cv.cvtColor(ideal, cv.COLOR_BGR2GRAY)
    kp_ideal, des_ideal = descriptor.detectAndCompute(gray_ideal, None)
    gray_ideal_rect = cv.polylines(cv.cvtColor(gray_ideal, cv.COLOR_GRAY2BGR), [np.int32(pts)], True, (255, 255, 0), 10,
                                   cv.LINE_AA)

    times = []
    shapes = []
    convex_metrics = []
    non_convex_metrics = []
    grouping = []
    grouping_ham = []

    for i, filename in enumerate(os.listdir(source_dir), start_numerate):
        image = cv.imread(os.path.join(source_dir, filename))
        if image is None:
            continue
        shapes.append(image.shape)

        start_time = time.time()
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        kp, des = descriptor.detectAndCompute(gray_image, None)
        try:
            matches = matcher.knnMatch(des_ideal, des, 2)
        except cv.error:
            print("error matching")
            print(i)
            dtime = (time.time() - start_time)
            times.append((i + 1, dtime))
            convex_metrics.append(0)
            non_convex_metrics.append(0)
            grouping.append(-1)
            grouping_ham.append(-1)
            continue
        dtime = (time.time() - start_time)
        times.append((i + 1, dtime))

        good = []
        for m, n in matches:
            if m.distance < dist_param * n.distance:
                good.append([m])
        # good = bf.match(des_ideal, des)

        dst = []
        dst_pts = []
        MIN_MATCH_COUNT = 10
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp_ideal[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, ransac_param)
            matchesMask = mask.ravel().tolist()

            dst = cv.perspectiveTransform(pts, M)

            gray_image = cv.polylines(cv.cvtColor(gray_image, cv.COLOR_GRAY2BGR), [np.int32(dst)], True,
                                      (255, 255, 0), 10, cv.LINE_AA)

        result = cv.drawMatchesKnn(gray_ideal_rect, kp_ideal, gray_image, kp, good, None,
                                   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor=(0, 255, 0))
        cv.imwrite(f'cv2.SIFT icona.jpg kisa result{i+1}.jpg', result)

        try:
            if M is not None:
                dst2 = cv.perspectiveTransform(src_pts, M)
                dist = []
                dist_ham = []
                for im, d in zip(dst2, dst_pts):
                    dist.append(np.linalg.norm(im - d))
                    dist_ham.append(1 - distance.hamming(im, d))
                grouping.append(np.mean(dist))
                grouping_ham.append(np.mean(dist_ham))
            else:
                grouping.append(-1)
                grouping_ham.append(-1)
        except UnboundLocalError:
            grouping.append(-1)
            grouping_ham.append(-1)

        Points = []
        for t in dst:
            Points.append((t[0][0], t[0][1]))
        # print(Points)
        poly2 = Polygon(Points)
        poly = poly2.convex_hull
        s = 0.
        s2 = 0.
        for t in dst_pts:
            p1 = Point((t[0][0], t[0][1]))
            s += poly.contains(p1)
            s2 += poly2.contains(p1)
        if len(dst_pts):
            convex_metrics.append(s / len(dst_pts))
            non_convex_metrics.append(s2 / len(dst_pts))
        else:
            convex_metrics.append(0)
            non_convex_metrics.append(0)

    metrics = pd.DataFrame({'number': map(lambda x: x[0], times),
                            'width': map(lambda x: x[1], shapes),
                            'height': map(lambda x: x[0], shapes),
                            'time': map(lambda x: np.round(x[1], 5), times),
                            'convex': map(lambda x: np.round(x, 5), convex_metrics),
                            'non-convex': map(lambda x: np.round(x, 5), non_convex_metrics),
                            'grouping': grouping,
                            'grouping_ham': grouping_ham})
    metrics.fillna(-1, inplace=True)
    metrics.to_csv(working_directory + '\\' + result_dir + '\\' + title + ' ' + ideal_name + ' ' + source_dir +
                   ' result.csv', sep='\t', header=True, index=False)


if __name__ == '__main__':
    Descriptor = cv.SIFT_create()
    Matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)
    Dist_param = 0.74
    Ransac_param = 15e-4
    Source_dir = 'Kisa'
    Result_dir = 'Result_kisa'
    Ideal_name = 'icona.jpg'
    experiment(descriptor=Descriptor,
               matcher=Matcher,
               source_dir=Source_dir,
               ideal_name=Ideal_name, result_dir=Result_dir,
               dist_param=Dist_param, ransac_param=Ransac_param, start_numerate=0)
