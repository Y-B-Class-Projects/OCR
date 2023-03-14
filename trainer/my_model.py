import os
import time
from collections import OrderedDict
import torch.nn.functional as F
import torch
from easyocr.model.model import Model
from easyocr.recognition import AlignCollate

from trainer import get_config
from utils import AttnLabelConverter


def load_model(saved_model, opt):
    if opt.rgb:
        opt.input_channel = 3
    converter = AttnLabelConverter(opt.character)
    num_class = len(converter.character)

    model = Model(input_channel=opt.input_channel, output_channel=opt.output_channel,
                  hidden_size=opt.hidden_size, num_class=num_class)
    another_dict = OrderedDict()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    original_dict = torch.load(saved_model, map_location=device)
    for k, v in original_dict.items():
        another_dict[k.replace('module.', '')] = v
    model.load_state_dict(another_dict)

    return model

def inference(model, Image, opt):
    start = time.time()
    converter = AttnLabelConverter(opt.character)
    align_collate = AlignCollate(imgH=opt.imgH,
                                 imgW=opt.imgW,
                                 keep_ratio_with_pad=opt.PAD,
                                 adjust_contrast=0)

    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        image_tensors, _ = align_collate([(Image, None)])
        batch_size = image_tensors.size(0)
        image = image_tensors.to(device)
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, preds_size)
        else:
            preds = model(image, text_for_pred, is_train=False)
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
        dashd_line = '-' * 80
        head = f'{"predicted_labels":25s}\tconfidence score'
        print(f'{dashd_line}\n{head}\n{dashd_line}')

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        prediction = None
        for pred, pred_max_prob in zip(preds_str, preds_max_prob):
            if 'Attn' in opt.Prediction:
                pred_EOS = pred.find('[s]')
                pred, pred_max_prob = pred[:pred_EOS], pred_max_prob[:pred_EOS]

                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                print(f'{pred:25s}\t{confidence_score:0.4f}')
                prediction = pred
                break
        stop = time.time()
        print(f'Inference time: {stop - start:.3f}s')
        return prediction

if __name__ == '__main__':
    opt = get_config("config_files/en_filtered_config.yaml")
    model = load_model("saved_models/en_filtered/best_accuracy.pth", opt)
    Image = "all_data/test/00000.png"
    prediction = inference(model, Image, opt)
    print(prediction)
