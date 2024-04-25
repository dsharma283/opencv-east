from imutils.object_detection import non_max_suppression
import os, sys, cv2, time, numpy as np
import argparse, tqdm


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_images', '-i', required=True,
                        help='the images that needs to be evaluated')
    parser.add_argument('--results', '-r', required=False, default='./results',
                        help='Output directory to save the detection output')
    parser.add_argument('--threshold', '-t', required=False, default=0.3, type=float,
                        help='confidence threshold for non max suppression')
    parser.add_argument('--box_threshold', '-b', required=False, default=0.6,
                        type=float, help='checkpoint file to resume from')
    parser.add_argument('--gt_path', '-g', required=True, default=False,
                        help='The ground truth text boxes to evaluate against')
    parser.add_argument('--viz', '-v', required=False, default=False, action='store_true',
                        help='generate the visual output on bounding boxes')
    return parser


def east_detect(image, viz=True):
    layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]
    orig = image.copy()
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    (H, W) = image.shape[:2]
    # set the new width and height and then determine the ratio in change
    # for both the width and height: Should be multiple of 32
    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)
    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]
    net = cv2.dnn.readNet("model/frozen_east_text_detection.pb")
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            # Set minimum confidence as required
            if scoresData[x] < 0.6:
                continue
            # compute the offset factor as our resulting feature maps will
            #  x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])


    boxes = non_max_suppression(np.array(rects), probs=confidences)
    # loop over the bounding boxes
    bbs = []
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        # draw the bounding box on the image
        if viz:
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
        bbs.append([startX, startY, endX, startY, endX, endY, startX, endY])
    #print(time.time() - start)
    return orig, bbs


def write_bb_file(boxes, fname):
    with open(fname, "a") as of:
        for box in boxes:
            line = ",".join(map(str, box))
            line += ",__\n"
            of.writelines(line)


def start_east(args):
    flist = os.listdir(args.input_images)
    flist = [item for item in flist if item.endswith('.jpg') or item.endswith('.png')]
    pbar = tqdm.tqdm(flist)
    for idx, im in enumerate(pbar):
        img = os.path.join(args.input_images, im)
        image = cv2.imread(img)
        out_image, boxes = east_detect(image, viz=args.viz)
        oimg = os.path.join(args.results, im.split('.')[0]+".txt")
        write_bb_file(boxes, oimg)
        if args.viz:
            oimg = os.path.join(args.results, "res_"+im)
            cv2.imwrite(oimg, out_image)


def start_main():
    args = process_args().parse_args()
    start_east(args)


if __name__ == "__main__":
    start_main()
