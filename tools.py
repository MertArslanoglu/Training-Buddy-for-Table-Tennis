import copy
import glob
import os
import numpy
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import sys

class tools2analyze():
    def __init__(self,label_dir, in_video_path):

        self.dir = label_dir
        self.in_video_path = in_video_path
        numpy.set_printoptions(threshold=sys.maxsize)

    def range_calculator(self, sequence, ID, current_frame, last_range):
        tahammul = 30
        flag = False
        if sequence.ndim == 1 and (current_frame - sequence[-1]) > tahammul:
            flag = True
            return None, flag
        if sequence.ndim == 1 and (current_frame - sequence[-1]) < tahammul + 1:
            return last_range, flag

        if (current_frame - sequence[-1][-1]) > tahammul:
            flag = True
            return None, flag
        sequence_shifted = numpy.vstack([[sequence[0]], sequence])
        difference = sequence[1:len(sequence)] - sequence_shifted[1:len(sequence_shifted) - 1]
        if sequence[-1][5] == sequence[-2][5]:
            print("debug needed")
        v, x = difference, sequence
        if sequence[-1][-1] == current_frame:
            range_coeff = 3 / v[-1][5]
        else:
            range_coeff = math.log(current_frame - sequence[-1][-1]) * 3
        if v[0][1] > 0: ###hız pozitif
            x_range_upper = x[-1][1] + range_coeff * (abs(v[-1][1]) + 0.01)
            x_range_lower = x[-1][1] - range_coeff * (abs(v[-1][1]) + 0.01)
        else:## hız negatif
            x_range_lower = x[-1][1] - range_coeff * (abs(v[-1][1]) + 0.01)
            x_range_upper = x[-1][1] + range_coeff * (abs(v[-1][1]) + 0.01)
        if v[0][2] > 0:  ###hız pozitif
            h_range_upper = x[-1][2] + range_coeff * (abs(v[-1][2]) + 0.01)
            h_range_lower = x[-1][2] - range_coeff * (abs(v[-1][2]) + 0.01)
        else:  ## hız negatif
            h_range_lower = x[-1][2] - range_coeff * (abs(v[-1][2]) + 0.01)
            h_range_upper = x[-1][2] + range_coeff * (abs(v[-1][2]) + 0.01)

        range = numpy.array([x_range_lower, x_range_upper, h_range_lower, h_range_upper, ID])
##alternatif range abs kullansın normal range yöne de baksın
        return range, flag

    def track(self, dict_coordinates):
        ranges = {} #ID : lower_x, upper_x, lower_y, upper_y
        dict_balls = {} #ID : status
        frames = {}
        ID = 0
        sequence_list ={} #for analyze parts
        ranges2plot = {}
        for fn in range(len(dict_coordinates)):
            frames[fn] = {}
            ranges2plot[fn + 1] = {}
            if dict_coordinates[fn] is None:
                for key, val in list(ranges.items()):
                    ranges[key], flag = self.range_calculator(dict_balls[key], key, fn, ranges[key])
                    ranges2plot[fn +1][key] = copy.deepcopy(ranges[key])
                    if flag:
                        #print((len(dict_balls[key]) < 4 and dict_balls[key].ndim >1) or (dict_balls[key].ndim  == 1))
                        if (len(dict_balls[key]) < 10 and dict_balls[key].ndim >1) or (dict_balls[key].ndim  == 1):###şimdiye kadar kaç tane gördüğümüze bakıp top olup olmadığını anlıyor
                            del dict_balls[key]
                            del sequence_list[key]
                        del ranges[key]
                        del ranges2plot[fn + 1][key]
                continue
            else: #top detect edildi
                status = []
                banned_IDs = []
                for crd in dict_coordinates[fn]:
                    status = "new"
                    coord = numpy.append(crd, fn)
                    if ranges: #active range varsa:
                        conf = -1
                        center = 0
                        for ball in ranges: #existing top mu diye bakılır
                            if ranges[ball] is not None:
                                rang = ranges[ball]
                                center = [(rang[0] + rang[1])/2, (rang[2] + rang[3])/2]
                                if ball not in banned_IDs:
                                    if (crd[1]>rang[0]) & (crd[1]<rang[1]) & (crd[2]>rang[2]) & (crd[2]<rang[3]): #eğer aktif range içindeyse:
                                        conf_holder, rang_holder = -abs(math.dist(center, [crd[1], crd[2]])), rang[4]
                                        if conf_holder > conf:
                                            conf = conf_holder
                                            status = rang_holder
                        banned_IDs.append(status)
                    if not status == "new":
                        dict_balls[status] = numpy.vstack([dict_balls[status], coord]) #update status of existing ball
                        frames[fn][status] = {}
                        frames[fn][status]["coord"] = coord
                        frames[fn][status]["type"] = None
                        sequence_list[status] = numpy.append(sequence_list[status],fn)
                        #ranges[status], flag = self.range_calculator(dict_balls[status], status, fn, ranges[status])

                    if status == "new":
                        dict_balls[ID] = coord #new ball detected
                        frames[fn][ID] = {}
                        frames[fn][ID]["coord"] = coord
                        frames[fn][ID]["type"] = None
                        ranges[ID] = np.array([coord[1]-0.1, coord[1] + 0.1, coord[2]-0.1, coord[2]+0.1, ID])
                        ranges2plot[fn + 1][ID] = copy.deepcopy(ranges[ID])
                        sequence_list[ID] = numpy.array([fn])
                        banned_IDs.append(ID)
                        ID += 1

                for key, val in list(ranges.items()):
                    ranges[key], flag = self.range_calculator(dict_balls[key], key, fn, ranges[key])
                    ranges2plot[fn + 1][key] = copy.deepcopy(ranges[key])
                    if flag:
                        if (len(dict_balls[key]) < 10 and dict_balls[key].ndim > 1) or dict_balls[key].ndim == 1:
                            del dict_balls[key]
                            del sequence_list[key]
                        del ranges[key]
                        del ranges2plot[fn + 1][key]
        for ball_num in dict_balls:
            frame_list2plot_sekis, sagadüstü, soladüstü= self.masadansekis(dict_balls[ball_num], sequence_list[ball_num], ball_num)
            frame_list2plot_vurus, sagdakivurdu, soldakivurdu = self.vurus(dict_balls[ball_num], sequence_list[ball_num], ball_num)
            frame_list2plot_firlatis, firlatiss = self.firlatis(dict_balls[ball_num],
                                                                           sequence_list[ball_num], ball_num)


            if len(sagadüstü) > 1:

                for sag in sagadüstü:
                    frames[sag][ball_num]["type"] = "sagadüstü"
            elif len(sagadüstü) > 0:

                frames[sagadüstü[0]][ball_num]["type"] = "sagadüstü"

            if len(soladüstü) > 1:
                for sol in soladüstü:
                    frames[sol][ball_num]["type"] = "soladüstü"
            elif len(soladüstü) > 0:
                frames[soladüstü[0]][ball_num]["type"] = "soladüstü"

            if len(sagdakivurdu) > 1:
                for sagvurdu in sagdakivurdu:
                    frames[sagvurdu][ball_num]["type"] = "sagdakivurdu"
            elif len(sagdakivurdu) > 0:
                frames[sagdakivurdu[0]][ball_num]["type"] = "sagdakivurdu"


            if len(soldakivurdu) > 1:
                for solvurdu in soldakivurdu:
                    frames[solvurdu][ball_num]["type"] = "soldakivurdu"
            elif len(soldakivurdu) > 0:
                frames[soldakivurdu[0]][ball_num]["type"] = "soldakivurdu"

            if len(firlatiss) > 1:
                for sol in soladüstü:
                    frames[sol][ball_num]["type"] = "firlatis"
            elif len(soladüstü) > 0:
                frames[soladüstü[0]][ball_num]["type"] = "firlatis"

        return frames, ranges2plot



    @property
    def labels(self):
        new = []
        lines = []

        coordinates = []
        x = []
        os.chdir(self.dir)  # labels path
        my_files = glob.glob('*.txt')  ##toplu text dosyalarını alfabetik sıra halibde arraye koyuyor
        for i in range(len(my_files)): new.append(int(my_files[i].split('_')[1].split('.')[0]))  ##sadece 119u aldı
        frame_list = sorted(new)  ##frmae numbera göre sıraladı
        dict_coordinates= {}
        dict_heights = {}
        dict_x = {}
        for frame_number in frame_list:
            f = open("D:\\YOLOv7\\yolov7-custom\\runs\\detect\\expow\\labels\\vidin_" + str(
                int(frame_number)) + ".txt", "r")
            check = []
            check.append(f.read().split())
                #print(np.array(check, dtype='float').reshape(-1,5))
            dict_coordinates[frame_number] = np.array(check, dtype='float').reshape(-1, 5)
            dict_heights[frame_number] = np.array(check, dtype='float').reshape(-1, 5)[:,2]
            dict_x[frame_number] = np.array(check, dtype='float').reshape(-1, 5)[:, 1]



        for i in range(frame_list[-1]):
            if not i in frame_list:
                dict_coordinates[i], dict_heights[i], dict_x[i] = None, None, None

        frame_list = np.array((frame_list))


        frame_mask = numpy.full((len(frame_list)), True)
        for index in range(len(frame_list)):
            if index > 5:
                if index < len(frame_list) - 5:
                    for i in [-5, 5]:
                        if abs(frame_list[index] - frame_list[index + i]) > 10:
                            frame_mask[index] = False

        return dict_coordinates

    def masadansekis(self, sequence, frame_list, ID):
        if sequence.ndim < 2:
            return [], [], []
        x = sequence[:,1]
        heights = sequence[:,2]
        sequence_shifted = numpy.vstack([[sequence[0]], sequence])
        difference = sequence_shifted[1:len(sequence_shifted) - 1] - sequence[1:len(sequence)]
        difference_shifted = numpy.vstack([[difference[0]], difference])
        second_diff = (difference_shifted[1:len(difference_shifted) - 1] - difference[1:len(difference)])
        second_diff_x = second_diff[:,1]
        second_diff_y = second_diff[:, 2]
        instances = np.logical_and((second_diff_y * 1000) < -15, heights[2:len(heights)] > 0.2)
        instances = np.logical_and(instances, (difference_shifted[1:len(difference_shifted)-1][:,2] * difference[1:len(difference)][:,2]) < 0)
        instances = np.logical_and(instances, abs(second_diff_x[0:len(second_diff_x)]) < 2)
        sagadüstü =np.array(frame_list[2:len(frame_list)][np.logical_and(instances, x[2:len(x)] > 0.5)])
        soladüstü= np.array(frame_list[2:len(frame_list)][np.logical_and(instances, x[2:len(x)] < 0.5)])
        frame_list2plot = np.array(frame_list[2:len(frame_list)][instances])

        return frame_list2plot, sagadüstü, soladüstü

    def vurus(self, sequence, frame_list, ID):
        if sequence.ndim <2:
            return [], [], []
        x = sequence[:, 1]
        sequence_shifted = numpy.vstack([[sequence[0]], sequence])
        difference = sequence_shifted[1:len(sequence_shifted) - 1] - sequence[1:len(sequence)]
        difference_shifted = numpy.vstack([[difference[0]], difference])
        second_diff = (difference_shifted[1:len(difference_shifted) - 1] - difference[1:len(difference)])
        second_diff_x = second_diff[:, 1]
        instance2 = np.logical_and(abs(second_diff_x * 1000) > 8, second_diff[0:len(second_diff)][:,2]>-12)
        sagdakivurdu = np.array(frame_list[2:len(frame_list)][np.logical_and(instance2, x[2:len(x)] > 0.5)])
        soldakivurdu = np.array(frame_list[2:len(frame_list)][ np.logical_and(instance2, x[2:len(x)] < 0.5)])
        frame_list2plot = np.array(frame_list[2:len(frame_list)][instance2])

        return frame_list2plot, sagdakivurdu, soldakivurdu


    def firlatis(self, sequence, frame_list, ID):
        if sequence.ndim <2:
            return [], [], []
        x = sequence[:, 1]
        sequence_shifted = numpy.vstack([[sequence[0]], sequence])
        difference = sequence_shifted[1:len(sequence_shifted) - 1] - sequence[1:len(sequence)]
        difference_x = difference[:, 1]
        difference_shifted = numpy.vstack([[difference[0]], difference])
        instance2 = np.logical_and((difference_x * 1000) > 4, x[1:len(x)] > 0.5)
        frame_list2plot = np.array(frame_list[1:len(frame_list)][instance2])
        firlatiss = np.array(frame_list[1:len(frame_list)][instance2])
        return frame_list2plot, firlatiss


    def video_out(self, frames, ranges2plot):
        final = list(frames)[-1]
        cap = cv2.VideoCapture(self.in_video_path + ".avi")
        fn = 0
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        video = cv2.VideoWriter(self.in_video_path + "_out.avi", fourcc, 60.0, (int(cap.get(3)), int(cap.get(4))))
        count=0
        atis=0
        atis_basari=0
        karsilama=0
        firlatis = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        karsilama_basari=0
        holder1 = -30 #soladüştü
        holder2 = -30 #sagadüştü
        holder3 = -30 #soldakivurdu
        holder4 = -30 #sagdakivurdu
        holder5 = -30 #firlatis
        holder7 = -30
        ball_pos = (0, 0)
        cnt_ball = 15
        ball_cnt = 0
        color_dict ={}
        ranges2plot[0] = {}
        while (cap.isOpened()):
            if fn == (final-1):
                break
            ranges = ranges2plot[fn]
            if cnt_ball > 10:
                ball_pos = (0, 0)
                cnt_ball = 0
            cnt_ball += 1
            ret, frame = cap.read()
            sizes = [frame.shape]
            sizes = [sizes[0][1], sizes[0][0]]
            if ret is True:
                if frames[fn] is not None:
                    for ID in frames[fn]:
                        if ID not in list(color_dict):
                            color_dict[ID] = (255, 0, 0)
                        detection = frames[fn][ID]
                        coordinates = detection["coord"]
                        if detection["type"] == "sagadüstü":
                            if (holder3 > fn - 70) and (holder3 < fn - 20):
                                if (fn > holder2 + 30):
                                    if (fn > holder7 + 30):
                                        print("sagadüstü")
                                        color_dict[ID] = (0, 255, 0)
                                        atis_basari = atis_basari + 1
                                        holder2 = fn
                                        ball_pos = (math.floor(sizes[0] * (float(coordinates[1]))),
                                                    math.floor(sizes[1] * (float(coordinates[2]))))
                        if detection["type"] == "soladüstü":
                            holder7 = fn
                            if (holder4 > fn - 35) and (holder4 < fn - 10):
                                if (fn > holder1 + 30):
                                    print("soladüstü")
                                    color_dict[ID] = (0, 255, 0)
                                    karsilama_basari = karsilama_basari + 1
                                    holder1 = fn
                                    ball_pos = (math.floor(sizes[0] * (float(coordinates[1]))),
                                                    math.floor(sizes[1] * (float(coordinates[2]))))

                        if detection["type"] == "soldakivurdu":
                            if fn > holder3 + 30:
                                print("soldakivurdu")
                                color_dict[ID] = (0, 0, 255)
                                atis = atis + 1
                                holder3 = fn


                        if detection["type"] == "sagdakivurdu":
                            if fn > holder4 + 30:
                                print("sagdakivurdu")
                                color_dict[ID] = (0, 0, 255)
                                karsilama = karsilama + 1
                                holder4 = fn

                        if detection["type"] == "firlatis":
                            if fn > holder5 + 30:
                                print("firlatis")
                                color_dict[ID] = (0, 127, 255)
                                firlatis = firlatis + 1
                                holder5 = fn

                        cv2.rectangle(frame, (
                        math.floor(sizes[0] * (float(coordinates[1]) - float(coordinates[3]) / 2)),
                        math.floor(sizes[1] * (float(coordinates[2]) - float(coordinates[4]) / 2))), (
                                      math.floor(sizes[0] * (float(coordinates[1]) + float(coordinates[3]) / 2)),
                                      math.floor(sizes[1] * (float(coordinates[2]) + float(coordinates[4]) / 2))),
                                      color_dict[ID], 2)

                        cv2.putText(frame, "ball_id:" + str(ID), (
                        math.floor(sizes[0] * (float(coordinates[1]) + float(coordinates[3]) / 2)),
                        math.floor(sizes[1] * (float(coordinates[2]) + float(coordinates[4]) / 2))), font, 0.5, (255, 255, 0),
                                    2,
                                    cv2.LINE_AA)



                #cv2.putText(frame, "basarili karsilama:" + str(karsilama_basari) + '/' + str(karsilama),
                                    #(10, 200), font, 3, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, "succes_rate:" + str(atis_basari) + '/' + str(firlatis), (10, 80), font, 3,
                                    (255, 255, 0), 2,
                                    cv2.LINE_AA)
                for keyy in ranges:
                    cv2.rectangle(frame, (
                        math.floor(sizes[0] * (ranges[keyy][0])),
                        math.floor(sizes[1] * (ranges[keyy][2]))), (
                                      math.floor(sizes[0] * (ranges[keyy][1])),
                                      math.floor(sizes[1] * (ranges[keyy][3]))),
                                  (0, 0, 0), 2)
                    if ranges[keyy][4] == 2:
                        print("amm")
                    cv2.putText(frame, "range_id:" + str(ranges[keyy][4]), (
                        math.floor(sizes[0] * (float(ranges[keyy][0]))),
                        math.floor(sizes[1] * (float(ranges[keyy][3])) + 10)), font, 2, (255, 255, 255),
                                2,
                                cv2.LINE_AA)


                video.write(frame)

                fn += 1
                print(fn)
            else:
                 break
        cap.release()
        video.release()
        cv2.destroyAllWindows()


