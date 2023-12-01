import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

# Initialize CLIP model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

def get_text_prompt_similarities(image_path, text_prompts):
    """
    For a given image, find the similarity for each text prompt.
    """
    image = Image.open(image_path).convert("RGB")
    image_input = processor(images=image, return_tensors="pt")["pixel_values"].to(device)

    prompt_similarities = {}

    for prompt in text_prompts:
        text_inputs = processor(text=prompt, return_tensors="pt", padding=True)["input_ids"].to(device)
        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=image_input).detach()
            text_features = model.get_text_features(input_ids=text_inputs).detach()

            similarity = torch.cosine_similarity(text_features, image_features).cpu().numpy()[0]
            prompt_similarities[prompt] = similarity

    return prompt_similarities

def main():
    image_path = "./negative/negative_000002.jpg"  # Replace with your image path
    text_prompts = ["a brick", "a dog", "a cat", "a tree", "a car", "a flower", "red cabinet"]  # Add more prompts as needed

    prompt_similarities = get_text_prompt_similarities(image_path, text_prompts)

    # Print the similarity for each prompt
    for prompt, similarity in prompt_similarities.items():
        print(f"Similarity between '{prompt}' and image: {similarity:.2f}")

    # Optionally, find the prompt with the highest similarity
    best_prompt = max(prompt_similarities, key=prompt_similarities.get)
    best_similarity = prompt_similarities[best_prompt]
    print(f"\nBest prompt: '{best_prompt}' with similarity: {best_similarity:.2f}")

if __name__ == "__main__":
    main()
