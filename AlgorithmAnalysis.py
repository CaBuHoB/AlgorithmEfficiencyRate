from os import path
import csv


# box is [upper left corner x, upper left corner y, width, height]
def bb_intersection_over_union(boxA, boxB):
    boxA[2] = boxA[0] + boxA[2]
    boxA[3] = boxA[1] + boxA[3]
    boxB[2] = boxB[0] + boxB[2]
    boxB[3] = boxB[1] + boxB[3]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def analysis(videos, annotations_path, path_with_the_results_of_the_algorithm):
    csvfile = open(path.join(path_with_the_results_of_the_algorithm, 'video_files__analysis.csv'), 'w')
    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['Video', 'IoU', 'GT obj', 'Detected obj', 'Corr det', 'Corr rec'])

    for video in videos:
        pos_annot = open(path.join(annotations_path, "pos_annot_" + path.splitext(video)[0] + ".txt"))
        result_alg = open(path.join(path_with_the_results_of_the_algorithm, "det_" + path.splitext(video)[0] + ".txt"))

        lol = lambda lst, sz: [[lst[i]] + [int(x) for x in lst[i + 1:i + sz]] for i in range(0, len(lst), sz)]

        col_frame = 0
        sum_IoU = 0.0
        sum_GT_obj = 0
        sum_detected_obj = 0
        sum_corr_det = 0
        sum_corr_rec = 0

        while True:
            boxesA = pos_annot.readline().split()
            if boxesA == []:
                break
            boxesA = lol(boxesA[1:], 5)

            boxesB = result_alg.readline().split()
            if boxesB == []:
                break
            boxesB = lol(boxesB[1:], 5)

            IoU_in_the_frame = 0.0
            GT_obj_in_the_frame = len(boxesA)
            detected_obj_in_the_frame = len(boxesB)
            corr_det_in_the_frame = 0
            corr_rec_in_the_frame = 0

            for boxA in boxesA:
                for boxB in boxesB:
                    IoU_boxA_boxB = bb_intersection_over_union(boxA[1:], boxB[1:])
                    if IoU_boxA_boxB > 0.4:
                        corr_det_in_the_frame = corr_det_in_the_frame + 1
                        IoU_in_the_frame = IoU_in_the_frame + IoU_boxA_boxB
                        if boxA[0] == boxB[0]:
                            corr_rec_in_the_frame = corr_rec_in_the_frame + 1
                        break

            col_frame = col_frame + 1
            sum_IoU = sum_IoU + (IoU_in_the_frame / corr_det_in_the_frame)
            sum_GT_obj = sum_GT_obj + GT_obj_in_the_frame
            sum_detected_obj = sum_detected_obj + detected_obj_in_the_frame
            sum_corr_det = sum_corr_det + corr_det_in_the_frame
            sum_corr_rec = sum_corr_rec + corr_rec_in_the_frame

        IoU = sum_IoU / col_frame
        GT_obj = sum_GT_obj / col_frame
        detected_obj = sum_detected_obj / col_frame
        corr_det = sum_corr_det / col_frame
        corr_rec = sum_corr_rec / col_frame

        filewriter.writerow([video, IoU, round(GT_obj), round(detected_obj), round(corr_det), round(corr_rec)])
