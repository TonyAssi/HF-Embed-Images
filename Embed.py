from transformers import AutoFeatureExtractor, AutoModel
from datasets import load_dataset
from huggingface_hub import create_repo


def extract_embeddings(image, extractor, model):
    image_pp = extractor(image, return_tensors="pt")
    features = model(**image_pp).last_hidden_state[:, 0].detach().numpy()
    return features.squeeze()


def create_dataset_embeddings(input_dataset, output_dataset, token, model_ckpt='google/vit-base-patch16-224'):
	# Load model for computing embeddings of the candidate images
	extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
	model = AutoModel.from_pretrained(model_ckpt)
	hidden_dim = model.config.hidden_size

	# Load dataset
	dataset = load_dataset(input_dataset, split="train")

	# Extract embeddings
	dataset_with_embeddings = dataset.map(lambda example: {'embeddings': extract_embeddings(example["image"].convert('RGB'), extractor, model)})

	# Create dataset repo
	create_repo(output_dataset, token=token, repo_type="dataset")

	# Push to hub
	dataset_with_embeddings.push_to_hub(output_dataset, token=token)