import torch
import json
from time import time, sleep
from helpers import get_predict_input_args, process_image
from model import load_model
from PIL import Image

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def main():

    in_arg = get_predict_input_args()

    predict(in_arg.image_path,
            in_arg.checkpoint,
            in_arg.topk,
            in_arg.category_names,
            in_arg.gpu)


# predict in image and print the topk classes predicted 
def predict(image_path, checkpoint, topk, cat_names, gpu):
    start_time = time()

    device = torch.device(gpu if torch.cuda.is_available() else "cpu")

    model = load_model(None, checkpoint, None).double()

    im = Image.open(image_path)

    processed_image = process_image(im)

    test_image = processed_image.unsqueeze(0)

    # predict image
    with torch.set_grad_enabled(False):
        test_image.to(device)
        logps = model.forward(test_image)
        ps = torch.exp(logps)
        top_p, top_classes = ps.topk(topk, dim=1)  

    idx_to_class = dict()
    # inverting dic
    for key, value in model.class_to_idx.items(): 
        idx_to_class[value] = key
    
    # printing results
    for p, c in zip(top_p[0].tolist(),top_classes[0].tolist()):
        print(f"class name: {cat_to_name[idx_to_class[c]]}"
              f"\t with probability of: {p * 100 : .3f}%")

    end_time = time()
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )


if __name__ == "__main__":
    main()