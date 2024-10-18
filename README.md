
# YOLOv5 Object Detection and Save

This project demonstrates how to use YOLOv5 to perform object detection on images and save the results. The script `yolov5-detect-and-save.py` allows users to load a YOLOv5 model, perform inference on an image, filter detections based on target classes, draw bounding boxes around detected objects, and save the processed image.

## Features

- Load a pre-trained YOLOv5 model (YOLOv5s).
- Perform object detection on input images.
- Filter detected objects by specified target classes (e.g., "person" and "dog").
- Draw bounding boxes and labels for detected objects.
- Save the processed image with detected objects highlighted.
- User interaction for processing multiple images.

## Requirements

- Python 3.6 or later
- PyTorch
- OpenCV
- YOLOv5

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/yolov5-detect-and-save.git
   cd yolov5-detect-and-save
   ```

2. Install the required packages:

   ```bash
   pip install torch opencv-python numpy
   ```

3. Make sure you have YOLOv5 installed. If not, you can install it using:

   ```bash
   pip install ultralytics
   ```

## Usage

1. Run the script:

   ```bash
   python yolov5-detect-and-save.py
   ```

2. When prompted, enter the path to the image file you want to process.

3. The script will perform object detection and save the output image with detected objects highlighted.

4. The processed image will be saved in the same directory as the input image, with `_detected` appended to the file name.

## Example

To detect objects in an image named `sample.jpg`, place the image in the project directory and run the script. After processing, you will find a new file named `sample_detected.jpg` with the detected objects.

## Configuration

The script allows customization of the target classes to detect. By default, it looks for "person" and "dog". You can change the target classes in the script by modifying the `target_classes` list.

## Troubleshooting

If you encounter permission issues when running on WSL, ensure that the target directories have the correct write permissions. You can change the permissions using:

```bash
chmod -R 777 /path/to/your/directory
```

If the YOLOv5 model fails to load, try setting `force_reload=True` when loading the model.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## About the Developer

For more information about the developer and other projects, please visit [murasan-net.com](https://murasan-net.com/).
