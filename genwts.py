import torch
import os
import struct
from repvgg import get_RepVGG_func_by_name, repvgg_model_convert


cls_names = os.listdir("./handpose_x_gesture_v1")

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

image = torch.ones(1, 3, 224, 224)
if torch.cuda.is_available():
    model.cuda()
    image = image.cuda()

model.eval()
print(model)
print('image shape ', image.shape)
preds = model(image)

os.makedirs("./wts", exist_ok=True)

f = open(f"./wts/{model_type}.wts", 'w')
f.write("{}\n".format(len(model.state_dict().keys())))
for k, v in model.state_dict().items():
    print('key: ', k)
    print('value: ', v.shape)
    vr = v.reshape(-1).cpu().numpy()
    f.write("{} {}".format(k, len(vr)))
    for vv in vr:
        f.write(" ")
        f.write(struct.pack(">f", float(vv)).hex())
    f.write("\n")

