from datasets import load_dataset
import matplotlib.pyplot as plt
from PIL import Image

# Load a dataset 
imagenette = load_dataset(
    'frgfm/imagenette',
    'full_size',
    split = 'train',
    ignore_verifications = False
)

# # Check if the dataset is loaded and has image data
# if 'image' in imagenette[0]:
#     plt.imshow(imagenette[0]['image'])
#     plt.show()
# else:
#     print("No image data found in the dataset.")


from transformers import CLIPProcessor, CLIPModel, CLIPTextConfig
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id).to(device)

tokenizer = CLIPProcessor.from_pretrained(model_id)

processor = CLIPProcessor.from_pretrained(model_id)


prompt = "A dog in a snow"

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print(inputs)

text_emb = model.get_text_features(**inputs)

print(text_emb.shape)

image = processor(
    text = None, 
    images = imagenette[0]['image'],
    return_tensors = "pt"
)['pixel_values'].to(device)

# print(image.shape)


image_emb = model.get_image_features(pixel_values = image)
print(image_emb.shape)
# plt.imshow(image.squeeze(0).T)
import numpy as np
np.random.seed(0)

sample_idx = np.random.randint(0, len(imagenette)+1, 100).tolist()
images = [imagenette[i]['image'] for i in sample_idx]

print(len(images))
image_array = None
batch_size = 16
from tqdm.auto import tqdm

for i in tqdm(range(0, len(images), batch_size)):
    batch = images[i:i+batch_size]
    batch = processor(
        text = None, 
        images = batch,
        return_tensors = "pt"
    )['pixel_values'].to(device)
    
    batch_emb = model.get_image_features(pixel_values = batch)
    batch_emb = batch_emb.squeeze(0)
    batch_emb = batch_emb.cpu().detach().numpy()
    if image_array is None:
        image_array = batch_emb
    else:
        image_array = np.concatenate((image_array, batch_emb), axis = 0)
    
print(image_array.shape)

print(image_array.min, image_array.max)



image_array = image_array / np.linalg.norm(image_array, axis = 1, keepdims = True)

text_emb = text_emb.cpu().detach().numpy()

scores = np.dot(text_emb, image_array.T)

print(scores.shape)

topK = 5
idx = np.argsort(scores)[0][::-1][:topK]

print(idx)

for i in idx:
    
    print(scores[0][i])
    plt.imshow(images[i])
    plt.show()
    