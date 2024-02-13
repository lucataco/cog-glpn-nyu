# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import torch
import numpy as np
from PIL import Image
from transformers import GLPNImageProcessor, GLPNForDepthEstimation

MODEL_NAME = "vinvino02/glpn-nyu"
MODEL_CACHE = "checkpoints"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.processor = GLPNImageProcessor.from_pretrained(
            MODEL_NAME,
            cache_dir=MODEL_CACHE)
        self.model = GLPNForDepthEstimation.from_pretrained(
            MODEL_NAME,
            cache_dir=MODEL_CACHE)


    def predict(
        self,
        image: Path = Input(description="Input image"),
    ) -> Path:
        """Run a single prediction on the model"""
        # prepare image for the model
        image = Image.open(image).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        # visualize the prediction
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth = Image.fromarray(formatted)

        output_path = "output.jpg"
        depth.save(output_path)
        return Path(output_path)









