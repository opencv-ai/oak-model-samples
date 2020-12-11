## Age-Gender_Recognition-Retail-0013

This is an inference code to run an age, gender recognition model on DepthAI using Gen2 Pipeline Builder.

The original model could be found in Intel Open Model Zoo, and the model card is [there](https://docs.openvinotoolkit.org/2019_R1/_age_gender_recognition_retail_0013_description_age_gender_recognition_retail_0013.html).

## Demo

![](demo.gif)

## Installation

1. Purchase a DepthAI model (see [shop.luxonis.com](https://shop.luxonis.com/))
2. Setup a package `python setup.py install` (use `python setup.py develop` for development installation)

## Usage

```
usage: inference.py [-h] [-cam] [-vid VIDEO] [-vis] [-cs WIDTHxHEIGHT]

optional arguments:
  -h, --help            show this help message and exit
  -cam, --camera        Use DepthAI RGB camera for inference (conflicts with
                        -vid)
  -vid VIDEO, --video VIDEO
                        Path to file to run inference on (conflicts with -cam)
  -vis, --visualization
                        Visualize the results from the network (required for
                        -cam)
  -cs WIDTHxHEIGHT, --capture-size WIDTHxHEIGHT
                        Frame shapes to capture with DepthAI RGB camera in WxH
                        format. The preview window will have the same shapes
                        (excluding legend).


```

To use with a video file, run the script with the following arguments:

```
python3 inference.py -vid ./demo.mp4
```

After the video is proceeded, frame-by-frame result will be stored in `inference_results.json`

You can also use `python3 inference.py -vid ./demo.mp4 -vis` for visualization.

To use with DepthAI 4K RGB camera, use instead:

```
python3 main.py -cam -vis
```

Also, you can specify the preview size with `-cs` key from the following:

```
["300x300", "640x480", "1280x720", "1920x1080"]
```

Lower resolution leads to higher inference speed.