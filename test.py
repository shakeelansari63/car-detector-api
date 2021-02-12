from car_detection import predict
import os
import multiprocessing as mp

files_prediction = mp.Manager().dict()
jobs = []


def img_predict(files_pred, img):
    prediction = predict(img)
    files_pred[img] = prediction


for fl in os.listdir('test_prediction'):
    if fl.endswith('jpg') or fl.endswith('jpeg'):
        img = os.path.join(os.path.abspath('test_prediction'), fl)

        p = mp.Process(target=img_predict, args=[files_prediction, img])
        p.start()

        jobs.append(p)

for job in jobs:
    job.join()

for img, prediction in files_prediction.items():
    print(img, ' - ', prediction)
