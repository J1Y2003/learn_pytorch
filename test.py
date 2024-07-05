import torch

def check_cuda():
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        # Get the number of available CUDA devices
        num_devices = torch.cuda.device_count()
        print(f"Number of CUDA Devices: {num_devices}")

        for i in range(num_devices):
            # Get the name of the CUDA device
            device_name = torch.cuda.get_device_name(i)
            print(f"CUDA Device {i}: {device_name}")

        # Perform a simple tensor operation on the GPU
        try:
            device = torch.device("cuda")
            x = torch.tensor([1.0, 2.0, 3.0], device=device)
            y = torch.tensor([4.0, 5.0, 6.0], device=device)
            z = x + y
            print(f"Tensor operation result: {z}")
        except Exception as e:
            print(f"An error occurred while performing tensor operations on the GPU: {e}")

if __name__ == "__main__":
    check_cuda()

print("Hi from PC!")