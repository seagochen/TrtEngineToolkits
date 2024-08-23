To enable and manage GPU devices on an NVIDIA Jetson device, you typically don't need to manually "enable" the GPUs, as they are already enabled and configured out of the box. However, there are a few things you might want to check or configure to ensure you're utilizing all GPU cores effectively, depending on your use case.

### 1. **Check the GPU Status**

First, you can check the status of the GPU using the following command:

```bash
tegra_stats
```

This command provides real-time information about the GPU utilization, CPU load, memory usage, and other statistics.

Alternatively, you can use the `jtop` tool (as installed previously) to monitor the GPU in real-time.

### 2. **Ensure Proper Power Mode**

Jetson devices have different power modes that can affect the performance of the GPU. To get the most out of your GPU, you should ensure that you are using a power mode that fully utilizes the GPU.

Check the current power mode:

```bash
sudo nvpmodel -q
```

To set the Jetson to maximum performance mode (which enables all CPU cores and maximizes GPU frequency), you can use:

```bash
sudo nvpmodel -m 0
```

Here, `-m 0` sets the device to maximum performance mode. Other modes may limit GPU and CPU performance to save power.

### 3. **Enable Jetson Clocks**

To further ensure that the GPU is operating at its maximum capability, you can enable `jetson_clocks`, which maximizes the frequency of the CPU, GPU, and other components:

```bash
sudo jetson_clocks
```

This command sets the clocks of the CPU, GPU, and other components to their maximum frequencies.

### 4. **Using GPU Acceleration in Applications**

To ensure that your applications utilize the GPU, you need to make sure that they are configured to use CUDA (for general-purpose GPU computing) or TensorRT (for optimized deep learning inference).

- **CUDA:** Make sure your applications are compiled with CUDA support and that the appropriate CUDA version is installed.
  
- **TensorRT:** If you’re deploying AI models, make sure to use TensorRT for optimized performance on Jetson devices.

### 5. **Check the GPU with CUDA Samples**

You can run CUDA samples to test if your GPU is working properly:

```bash
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
```

The output should show the properties of the GPU and confirm that it is functioning properly.

### Summary

By ensuring that your Jetson device is in the correct power mode, running `jetson_clocks`, and making sure your applications are configured to use GPU acceleration, you can fully utilize all GPU resources on your Jetson device.
