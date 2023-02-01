import torch
import cv2
import numpy as np
from repvgg import get_RepVGG_func_by_name, repvgg_model_convert

cls_names = ['000-one', '001-five', '002-fist', '003-ok', '004-heartSingle', '005-yearh', '006-three', '007-four',
            '008-six', '009-Iloveyou', '010-gun', '011-thumbUp', '012-nine', '013-pink']

def softmax_T(X, T=1):
    """
    Calculate the softmax function (Tempoarl T).

    The softmax function is calculated by
    np.exp(X/T) / np.sum(np.exp(X/T), axis=1)

    This will cause overflow when large values are exponentiated.
    Hence the largest value in each row is subtracted from each data
    point to prevent this.

    Parameters
    ----------
    X : array-like, shape (M,)
        Argument to the logistic function

    copy : bool, optional
        Copy X or not.

    Returns
    -------
    out : array, shape (M,)
        Softmax function evaluated at every point in x
    """

    X = X/T
    max_prob = np.max(X)
    X -= max_prob
    X = np.exp(X)
    sum_prob = np.sum(X)
    X /= sum_prob
    return X

cls_nums = len(cls_names)

model_type = "RepVGG-A0"

train_model = get_RepVGG_func_by_name(model_type)(deploy=False)
fc_input_nums = train_model.linear.in_features
train_model.linear = torch.nn.Linear(in_features=fc_input_nums, out_features=cls_nums, bias=True)

cpkt = torch.load(f"./ckpt/checkpoint_best.pth.tar", map_location=torch.device('cpu'))[
    'state_dict']
new_weights = {}
for k, v in cpkt.items():
    if "module" in k:
        new_weights[k[7:]] = v
    else:
        new_weights[k] = v
train_model.load_state_dict(new_weights)

model = repvgg_model_convert(train_model)

model.eval()
path = f"./data/val/000-one/gesture-one-2021-03-07_23-07-48-1_37388.jpg"

image_o = cv2.imread(path)
image = image_o[..., ::-1]


img = cv2.resize(image,(224,224))
img = img/255
img -= 0.5
img /= 0.5

img = torch.Tensor(img)
img = img.permute((2, 0, 1))
img = torch.unsqueeze(img, dim=0)

out = model(img)
out = softmax_T(out.detach().numpy()[0], T=1)
print(out)

cls_res = np.argmax(out)
print(out[cls_res])
pre = cls_names[cls_res]
print(pre)
cv2.imshow("!", image_o)
cv2.waitKey(0)
