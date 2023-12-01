import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import matplotlib.pyplot as plt

# Initialize CLIP model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

def load_images_from_folder(folder):
    """
    Load all images from the given folder.
    """
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img = Image.open(os.path.join(folder, filename)).convert("RGB")
            images.append(img)
            filenames.append(filename)
    return images, filenames

def get_similarity(images, text):
    """
    Calculate the similarity between images and a text description.
    """
    text_inputs = processor(text=text, return_tensors="pt", padding=True)["input_ids"].to(device)
    text_features = model.get_text_features(**{"input_ids": text_inputs}).detach()

    similarities = []
    for image in images:
        image_inputs = processor(images=image, return_tensors="pt", padding=True)["pixel_values"].to(device)
        image_features = model.get_image_features(**{"pixel_values": image_inputs}).detach()

        # Calculate cosine similarity
        similarity = torch.cosine_similarity(text_features, image_features).cpu().numpy()[0]
        similarities.append(similarity)

    return similarities
def plot_images_with_similarity(images, filenames, similarities, text):
    """
    Plot each image with its similarity score.
    """
    plt.figure(figsize=(20, 10))
    columns = min(5, len(images))
    for i, (image, filename, similarity) in enumerate(zip(images, filenames, similarities)):
        plt.subplot(len(images) // columns + 1, columns, i + 1)
        plt.imshow(image)
        plt.title(f"{filename}\nSimilarity: {similarity:.2f}")
        plt.axis('off')

    plt.suptitle(f"Similarity to '{text}'", fontsize=16)
    plt.show()

def main():
    folder_path = "./negative"  # Replace with your images folder path
    images, filenames = load_images_from_folder(folder_path)

    prompt = "Red Cabinet"
    similarities = get_similarity(images, prompt)
    for filename, similarity in zip(filenames, similarities):
            print(f"Similarity between '{prompt}' and '{filename}': {similarity}")
    plot_images_with_similarity(images, filenames, similarities, prompt)

if __name__ == "__main__":
    main()