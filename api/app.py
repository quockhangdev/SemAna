import litserve as ls
import random
from pydantic import BaseModel
random.seed(42)

class InputRequest(BaseModel):
    text: str

class OutputResponse(BaseModel):
    label: str

class SemAnaAPI(ls.LitAPI):
    def setup(self, device):
        self.model = lambda x: random.uniform(0, 1)
        
    def decode_request(self, request: InputRequest, context):
        return request.text
    
    def predict(self, x):
        prediction = self.model(x)
        output = "positive" if prediction > 0.5 else "negative"
        return output
    
    def encode_response(self, output, context):
        return OutputResponse(label=output)
    
if __name__ == "__main__":
    server = ls.LitServer(SemAnaAPI(), api_path="/classify", accelerator="auto", max_batch_size=1)
    server.run(port=8000)
