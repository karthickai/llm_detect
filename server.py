from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from litserve import LitAPI, LitServer

class DistilBertAPI(LitAPI):
    def setup(self, device):
        """
        Load the tokenizer and model, and move the model to the specified device.
        """
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained("models/distilbert-llm-detect")
        self.model.to(device)
        self.model.eval()
    
    def decode_request(self, request):
        """
        Preprocess the request data (tokenize)
        """
        # Assuming request is a dictionary with a "text" field
        inputs = self.tokenizer(request["text"], return_tensors="pt", padding=True, truncation=True, max_length=512)
        return inputs

    def predict(self, inputs):
        """
        Perform the inference
        """
        with torch.no_grad():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
        return outputs.logits
    
    def encode_response(self, logits):
        """
        Process the model output into a response dictionary
        """
        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        prediction = torch.argmax(probabilities, dim=-1).item()
        
        response = {
            "status": "Prompt Injection" if prediction == 1 else "No Prompt Injection",
        }
        return response

if __name__ == "__main__":
    api = DistilBertAPI()
    server = LitServer(api, accelerator='cuda', devices=1)
    server.run(port=8000)