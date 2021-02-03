## Installation

1. Purchase a DepthAI model (see [shop.luxonis.com](https://shop.luxonis.com/))
2. Set up a package:

   2.1 `python3 setup.py bdist_wheel && rm -R build/ *.egg-info`

   2.2 `pip3 install dist/*.whl -f https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/depthai/ && rm -R dist/`

## Usage

```
usage: main.py [-h] [-cam] [-vid VIDEO] [-vis] [-cs WIDTHxHEIGHT]
               [--threshold THRESHOLD] [--save-results]

optional arguments:
  -h, --help            show this help message and exit
  -cam, --camera        Use DepthAI RGB camera for inference (conflicts with
                        -vid)
  -vid VIDEO, --video VIDEO
                        Path to file to run inference on (conflicts with -cam)
  -vis, --visualization
                        Visualize the results from the network (always on for
                        camera)
  --threshold THRESHOLD, -tr THRESHOLD
                        Threshold for model predictions
  --save-results, -sr   Save by-frame results of the inference into json

```

To use with a video file, run the script with the following arguments:

```
python3 main.py -vid ./demo.mp4
```

After the video is proceeded, frame-by-frame result will be stored in `inference_results.json` (if `--save_results` key is specified)

You can also use `python3 main.py -vid ./demo.mp4 -vis` for visualization.

To use with DepthAI 4K RGB camera, use instead:

```
python3 main.py -cam -vis
```
