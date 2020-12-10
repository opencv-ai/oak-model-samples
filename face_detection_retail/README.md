## Face-Detection-Retail-0004

This is an inference code to run a face detection model on DepthAI using Gen2 Pipeline Builder.

The original model could be found in Intel Open Model Zoo, and the model card is [there](https://docs.openvinotoolkit.org/2018_R5/_docs_Retail_object_detection_face_sqnet10modif_ssd_0004_caffe_desc_face_detection_retail_0004.html).

## Demo

![](demo.gif)

## Installation

1. Purchase a DepthAI model (see [shop.luxonis.com](https://shop.luxonis.com/))
2. Setup a package `python setup.py install` (use `python setup.py develop` for development installation)

## Usage

```
usage: inference.py [-h] [-cam] [-vid VIDEO] [-vis]

optional arguments:
  -h, --help            show this help message and exit
  -cam, --camera        Use DepthAI RGB camera for inference (conflicts with -vid)
  -vid VIDEO, --video VIDEO
                        Path to file to run inference on (conflicts with -cam)
  -vis, --visualization
                        Visualize the results from the network (required for -cam)

```

To use with a video file, run the script with the following arguments

```
python3 inference.py -vid ./demo.mp4
```

After the video is proceeded, frame-by-frame result will be stored in `inference_results.json`

You can also use `python3 inference.py -vid ./demo.mp4 -vis` for visialization.

To use with DepthAI 4K RGB camera, use instead

```
python3 main.py -cam -vis
```
