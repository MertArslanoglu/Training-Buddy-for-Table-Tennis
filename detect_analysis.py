import argparse
import time
from pathlib import Path

import numpy
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import copy
from tools import tools2analyze
import math



def detect(save_img=False):
    dir = r'D:\\YOLOv7\\yolov7-custom\\runs\\detect\\expow\\labels'
    video_path = "D:\\YOLOv7\\yolov7-custom\\metalurji\\vidin"
    analyze = tools2analyze(dir, video_path)
    fn = -1
    dict_coordinates = {}
    source, weights, view_img, save_txt, imgsz, trace = "./metalurji/vidin.avi", "metalurjimodel.pt", False, True, 640, True
    save_img = True  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    #save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    save_dir = Path("runs/detect/expow")
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, 640)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    ranges = {}  # ID : lower_x, upper_x, lower_y, upper_y
    dict_balls = {}  # ID : status
    frames = {}
    ID = 0
    sequence_list = {}  # for analyze parts
    ranges2plot = {}
    circle_flag = True


    for path, img, im0s, vid_cap in dataset:



        fn = fn + 1
        coordinates = []
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=False)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=False)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, 0.2, 0.45, classes=None, agnostic=False)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if True:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if False else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        line2plot = [cls.cpu(), xywh[0], xywh[1], xywh[2], xywh[3]]
                        coordinates.append(line2plot)

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

        if coordinates:
            dict_coordinates[fn] = numpy.array(coordinates, dtype='float').reshape(-1, 5)
        else:
            dict_coordinates[fn] = None

        frames[fn] = {}
        ranges2plot[fn + 1] = {}
        if dict_coordinates[fn] is None:
            for key, val in list(ranges.items()):
                ranges[key], flag = analyze.range_calculator(dict_balls[key], key, fn, ranges[key])
                ranges2plot[fn + 1][key] = copy.deepcopy(ranges[key])
                if flag:
                    # print((len(dict_balls[key]) < 4 and dict_balls[key].ndim >1) or (dict_balls[key].ndim  == 1))
                    if (len(dict_balls[key]) < 10 and dict_balls[key].ndim > 1) or (dict_balls[
                                                                                        key].ndim == 1):  ###şimdiye kadar kaç tane gördüğümüze bakıp top olup olmadığını anlıyor
                        del dict_balls[key]
                        del sequence_list[key]
                    del ranges[key]
                    del ranges2plot[fn + 1][key]
            continue
        else:  # top detect edildi
            status = []
            banned_IDs = []
            for crd in dict_coordinates[fn]:
                status = "new"
                coord = numpy.append(crd, fn)
                if ranges:  # active range varsa:
                    conf = -1
                    center = 0
                    for ball in ranges:  # existing top mu diye bakılır
                        if ranges[ball] is not None:
                            rang = ranges[ball]
                            center = [(rang[0] + rang[1]) / 2, (rang[2] + rang[3]) / 2]
                            if ball not in banned_IDs:
                                if (crd[1] > rang[0]) & (crd[1] < rang[1]) & (crd[2] > rang[2]) & (
                                        crd[2] < rang[3]):  # eğer aktif range içindeyse:
                                    conf_holder, rang_holder = -abs(math.dist(center, [crd[1], crd[2]])), rang[4]
                                    if conf_holder > conf:
                                        conf = conf_holder
                                        status = rang_holder
                    banned_IDs.append(status)
                if not status == "new":
                    dict_balls[status] = numpy.vstack([dict_balls[status], coord])  # update status of existing ball
                    frames[fn][status] = {}
                    frames[fn][status]["coord"] = coord
                    frames[fn][status]["type"] = None
                    sequence_list[status] = numpy.append(sequence_list[status], fn)
                    # ranges[status], flag = self.range_calculator(dict_balls[status], status, fn, ranges[status])

                if status == "new":
                    dict_balls[ID] = coord  # new ball detected
                    frames[fn][ID] = {}
                    frames[fn][ID]["coord"] = coord
                    frames[fn][ID]["type"] = None
                    ranges[ID] = numpy.array([coord[1] - 0.1, coord[1] + 0.1, coord[2] - 0.1, coord[2] + 0.1, ID])
                    ranges2plot[fn + 1][ID] = copy.deepcopy(ranges[ID])
                    sequence_list[ID] = numpy.array([fn])
                    banned_IDs.append(ID)
                    ID += 1

            for key, val in list(ranges.items()):
                ranges[key], flag = analyze.range_calculator(dict_balls[key], key, fn, ranges[key])
                ranges2plot[fn + 1][key] = copy.deepcopy(ranges[key])
                if flag:
                    if (len(dict_balls[key]) < 10 and dict_balls[key].ndim > 1) or dict_balls[key].ndim == 1:
                        del dict_balls[key]
                        del sequence_list[key]
                    del ranges[key]
                    del ranges2plot[fn + 1][key]

    for ball_num in dict_balls:
        frame_list2plot_sekis, sagadüstü, soladüstü = analyze.masadansekis(dict_balls[ball_num], sequence_list[ball_num],
                                                                        ball_num)
        frame_list2plot_vurus, sagdakivurdu, soldakivurdu = analyze.vurus(dict_balls[ball_num], sequence_list[ball_num],
                                                                       ball_num)
        frame_list2plot_firlatis, firlatiss = analyze.firlatis(dict_balls[ball_num],
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
            for fir in firlatiss:
                frames[fir][ball_num]["type"] = "firlatis"
        elif len(firlatiss) > 0:
            frames[firlatiss[0]][ball_num]["type"] = "firlatis"

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

    analyze.video_out(frames, ranges2plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()

    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
            detect()
