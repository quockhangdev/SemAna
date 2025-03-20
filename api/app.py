import sys
sys.path.append(".")
import litserve as ls
import random
from pydantic import BaseModel
from predict import predict
from utils.model_utils import load_model
random.seed(42)

class InputRequest(BaseModel):
    text: str

class OutputResponse(BaseModel):
    label: str

class SemAnaAPI(ls.LitAPI):
    def setup(self, device):
        if device == "mps:0": # MPS is not supported
            device = "cpu"
        print("Loading model and tokenizer... using device: ", device)
        t, m = load_model(device=device)
        self.tokenizer = t
        self.model = m
        
    def decode_request(self, request: InputRequest, context):
        return request.text
    
    def predict(self, x):
        output = predict(self.model, self.tokenizer, x)
        return output
    
    def encode_response(self, output, context):
        return OutputResponse(label=output)
    
if __name__ == "__main__":
    server = ls.LitServer(SemAnaAPI(), api_path="/classify", accelerator="auto", max_batch_size=1)
    server.run(port=8000)
