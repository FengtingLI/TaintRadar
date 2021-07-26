# TaintRadar

Official Implementation of
[Detecting Localized Adversarial Examples: A Generic Approach using Critical Region Analysis](https://arxiv.org/abs/2102.05241)
published at IEEE International Conference on Computer Communications (INFOCOM) 2021.

## Installation

Please download the code:

To use our code, first download the repository:

````
git clone https://github.com/FengtingLI/TaintRadar.git
````

To install the dependencies:

````
conda create -n taint_radar -y python=3.7 tensorflow-gpu=1.13 keras=2.3.1
conda activate taint_radar
pip install -r requirements.txt
````

## Running

We provided an example of our method. The resources (model and images) can be found [here (google drive)](https://drive.google.com/drive/folders/1ChtzopluxJm-wxdPQlfsXJuwx3u0nOrF?usp=sharing). Download
the .zip files and extract them directly under the main folder, like this:

```
--> TaintRadar
    --> models
        --> vgg16.h5
    --> images
        --> *.png   # The attacked images
        --> *_origin.png # The corresponding original images
```

Then, run the following code:

```
python run.py
```

## Citation

If you find this code useful, please consider citing the following paper:

````
@inproceedings{li2021detecting,
  title     = {Detecting Localized Adversarial Examples: A Generic Approach using Critical Region Analysis},
  author    = {Fengting Li, Xuankai Liu, Xiaoli Zhang, Qi Li, Kun Sun, Kang Li},
  booktitle = {{IEEE} Conference on Computer Communications, {INFOCOM}},
  year      = {2021}
}
````
