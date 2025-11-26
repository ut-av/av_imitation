# Autonomous Vehicle Imitation Learning

This package has all the tools for training an imitation learning policy.

## Build and Install the Package

Build and install via symlink, so you don't need to re-build each time the webapp is changed:

```bash
colcon build --packages-select av_imitation --symlink-install
```

- First, record the data using the [av_recorder](https://github.com/ut-av/av_recorder) package.

## Data Cleaning

### Run Server

On the car, in the container, run

```bash
ros2 run av_imitation start_webapp
```

Then, using your computer, connect to the car. For example: [http://orin12:5000](http://orin12:5000)

### Clip the Bag

### Aggregate Bags into a Datset

### Visualize the Dataset

## Train a Model

## Convert to onnx

```bash
ros2 run av_imitation export_onnx --experiment-name 20251124_235857_mlp_outdoor_v1
```

## Run the onnx model on the car

```bash
ros2 run av_imitation inference --ros-args \
    -p model_path:=/path/to/model.onnx \
    -p image_topic:=/camera/image_raw \
    -p drive_topic:=/drive
```
