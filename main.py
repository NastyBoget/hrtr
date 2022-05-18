from doctr.io import DocumentFile
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models import detection_predictor
import os
import cv2


def test(model: DetectionPredictor, path_test: str = None):
    outdir = "/home/ox/work/datasets/test_doctr/"
    os.makedirs(outdir, exist_ok=True)
    files = os.listdir(path_test)
    cnt = len(files)
    for i, filename in enumerate(files):

        single_img_doc = DocumentFile.from_images(os.path.join(path_test, filename))
        #result = model(single_img_doc)
        #result.show(single_img_doc)
        out = model(single_img_doc)
        im = cv2.imread(os.path.join(path_test, filename))
        h, w, _ = im.shape

        for box in out[0]:
            cv2.rectangle(im, (int(box[0] * w), int(box[1] * h)), (int(box[2] * w), int(box[3] * h)),
                          color=(0, 255, 0), thickness=2)
            cv2.putText(im, "{0:0.2f}".format(box[4]), (int(box[0] * w), int(box[1] * h)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color=(255, 0, 0), thickness=1)

        cv2.imwrite(os.path.join(outdir, filename), im)
        print(f"{i}/{cnt} name={filename}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    '''model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True,
                          export_as_straight_boxes=True)'''
    model = detection_predictor(arch='db_resnet50_rotation', pretrained=True).eval()
    path1 = "/home/ox/work/datasets/podlozhka/licences/"
    path2 = "/home/ox/work/datasets/text_localization/napalm/train_images"
    test(model, path2)

