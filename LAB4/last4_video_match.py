import numpy as np
import cv2 as cv
import os
import time
from collections import Counter
import pandas as pd
from joblib import dump, load

def get_descriptor(img, descriptor):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kp, des = descriptor.detectAndCompute(gray, None)
    return kp, des


def init_cap(fname):
    cap = cv.VideoCapture(fname)
    width, height = (
        int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    )

    fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv.VideoWriter()
    out.open('my_test_game.mp4', fourcc, 24, (width, height), True)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return None
    else:
        return cap, out, fourcc, width, height


def match(des1, des2, matcher, alpha=0.62):
    matches = matcher.knnMatch(des1, des2, 2)

    good = []
    for m, n in matches:
        if m.distance < alpha * n.distance:
            good.append(m)
    return good

def maroon(des):
    return True

def count(data,clasterizator,N_cl=100):
    keys = range(0,N_cl)
    cnt = Counter(clasterizator.predict(data))
#     print(data.shape)
    t_freq = [(cnt[c]+0.)/len(data) for c in keys]
    return np.array(t_freq)


def pipeline(descriptor):
    result = []
    for i, el in enumerate(models):
        marked = count(descriptor, el[0], el[2])
        marked = np.array([marked])
        result.append(int(el[1].predict_proba(marked)[0][1] >= 0.65)  *  (i+1))
        print(el[1].predict_proba(marked))
    return result

def load_models(names, N_clusters, descriptor):
    models = []
    for name, N_cl in zip(names, N_clusters):
        cls = load(name+ '_' + descriptor + '_clusterizator_' + str(N_cl) + '.joblib')
        clf = load(name+ '_' + descriptor + '_classifier_' + str(N_cl) + '.joblib')
        models.append([cls, clf, N_cl])
    return models

# folder = 'Iryna game'
descriptor = cv.AKAZE_create()
matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_SL2)

ideal = cv.imread('irina_ideal_2.jpg')
victim = cv.imread('victim_ira_2.jpg')
victim = cv.resize(victim, (ideal.shape[1], ideal.shape[0]))

h, w, d = ideal.shape
pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

kp_ideal, des_ideal = get_descriptor(ideal, descriptor)
cap, out, fourcc, width, height = init_cap("test_game.mp4")
# cap = cv.VideoCapture('https://192.168.50.112:8080/video')
cap2, out2, fourcc2, width2, height2 = init_cap("COFFIN DANCE ORIGINAL VIDEO.mp4")

times = []

i = 0
flag = False
silly = False
if silly:
    predictor = maroon
else:
    model = load("game_AKAZE_classifier_25.joblib")
    clusterizator = load("game_AKAZE_clusterizator_25.joblib")
    N_clusters = 25
    models = [[clusterizator, model, N_clusters]]
    predictor = pipeline
print('--------------------------------------------------------')
while cap.isOpened():

    print('Q')
    start = time.time()
    ret, image = cap.read()

    if not ret:
        print('DEAD')

        finish = time.time()
        times.append(finish - start)

        break
    if image is None:

        finish = time.time()
        times.append(finish - start)

        continue
 # OR if flag
    r, victim = cap2.read()
    victim = cv.resize(victim, (ideal.shape[1], ideal.shape[0]))
 #    break
    kp, des = get_descriptor(image, descriptor)
    if not flag:
        flag = predictor(des)
        S_prev = -1
        finish = time.time()
        times.append(finish - start)
        continue
    else:
        good = match(des_ideal, des, matcher, alpha=0.6)

        MIN_MATCH_COUNT = 10
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp_ideal[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 2)
            # matchesMask = mask.ravel().tolist()

            if M is not None:
                dst = cv.perspectiveTransform(pts, M)
                vec1 = dst[2] - dst[0]
                vec2 = dst[3] - dst[1]
                # print(type(vec1))
                # print(vec1.shape)
                vec1 = np.hstack((np.zeros((1,1)),vec1))
                vec2 = np.hstack((np.zeros((1,1)), vec2))
                # print(vec1.shape)
                # print(vec2.shape)
                S = np.linalg.norm(np.cross(vec1,vec2))
                if max(S_prev/S, S/S_prev) > 1.5:
                    flag = False
                    out.write(image)
                    S_prev = S

                    finish = time.time()
                    times.append(finish - start)

                    continue
                S_prev = S
                img2 = cv.warpPerspective(victim, M, (width, height))
                black_frame = np.full((height, width), 255, np.uint8)
                cv.fillPoly(black_frame, [np.int32(dst)], (0, 0, 0))

                i += 1
                print(i)
                result_frame = cv.bitwise_and(image, image, mask=black_frame)
                result_image = cv.bitwise_or(img2, result_frame)

                out.write(result_image)
                result_image = cv.resize(result_image, (500,500))
                cv.imshow('frame',result_image)

                finish = time.time()
                times.append(finish - start)
            else:
                flag = False
                out.write(image)
                result_image = cv.resize(image, (500,500))
                cv.imshow('frame', result_image)

                finish = time.time()
                times.append(finish - start)
        else:
            flag = False
            out.write(image)
            result_image = cv.resize(image, (500, 500))
            cv.imshow('frame', result_image)

            finish = time.time()
            times.append(finish - start)
        # cv.imshow('frame', victim)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        # plt.imshow('frame', image)

times_by_frame = pd.DataFrame(times, columns=['Time'])
times_by_frame.to_csv('result.csv', sep='\t')

cap.release()
cap2.release()
out.release()
cv.destroyAllWindows()
