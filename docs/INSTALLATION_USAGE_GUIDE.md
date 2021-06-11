## Installation

## Pre-requirements

Make sure you have installed:

- Python3.7+
- [Git-LFS](https://docs.github.com/en/github/managing-large-files/installing-git-large-file-storage)

1. Purchase a DepthAI model (see [shop.luxonis.com](https://shop.luxonis.com/))
2. Set up a package:

   2.1 `pip3 install wheel && python3 setup.py bdist_wheel && rm -R build/ *.egg-info`

   2.2 `pip3 install dist/*.whl && rm -R dist/`

<details>
  <summary><b>Windows Installation</b></summary>

<p>

Set up a package:

1. `pip3 install wheel && python setup.py bdist_wheel`

2. `for /r %G in ("dist\*.whl") do pip3 install "%~G"`

3. `for /d %G in ("*.egg-info", "build\", "dist") do rd /s /q "%~G"`

</details>

## Usage

```
usage: main.py [-h] [-cam] [-vid VIDEO] [-vis] [--threshold THRESHOLD]
               [--save-results] [--preview_shape N N]

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
  --preview_shape W H, -ps W H
                        Shapes for preview windows. Higher resolution leads to
                        slower performance.


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

## Troubleshooting

<details>
  <summary><b>RuntimeError</b>: Failed to find device after booting, error message: <b>X_LINK_DEVICE_NOT_FOUND</b></summary>

  <p>

If while running the app, you get an error:

`Failed to find device after booting, error message: X_LINK_DEVICE_NOT_FOUND`

1. Run the following command:

   ```bash
   echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | \
   sudo tee /etc/udev/rules.d/80-movidius.rules && \
   sudo udevadm control --reload-rules
   ```

2. Unplug and replug an OAK

</details>
