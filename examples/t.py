import intel_npu_acceleration_library as npu_lib
import time  # Import time module for timing inference
print("step 1")
import torch
class NNModule(torch.nn.Module):
    def __init__(self):
        super(NNModule, self).__init__()
        layers = []
        in_channels = 256
        
        for _ in range(100):  # Create 100 layers of CNN
            layers.append(torch.nn.Conv2d(in_channels, 512, 3, 1, 1))  # Increased output channels
            layers.append(torch.nn.ReLU())
            in_channels = 512  # Update in_channels for the next layer
        
        self.conv_layers = torch.nn.Sequential(*layers)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc = torch.nn.Linear(512 * 16 * 16, 10)  # Adjusted input size for fully connected layer

    def forward(self, x):
        x = self.conv_layers(x)  # Pass through all convolutional layers
        x = self.pool(x)
        x = x.view(-1, 512 * 16 * 16)  # Adjusted for new output size
        x = self.fc(x)
        return x


model = NNModule()

print(next(model.parameters()).device)

x = torch.randn(1, 256, 32, 32)  # Generate random input tensor

# Time the inference on CPU
start_time = time.time()  # Start timing
_ = model(x)  # Inference
end_time = time.time()  # End timing

# Print the time taken for inference
print(f"Inference time on CPU: {end_time - start_time} seconds")

# Move model to NPU
model = model.to("npu")

# print the device
print(next(model.parameters()).device)

# Time the inference on NPU
start_time = time.time()  # Start timing
_ = model(x)  # Inference
end_time = time.time()  # End timing

# Print the time taken for inference
print(f"Inference time on NPU: {end_time - start_time} seconds")
