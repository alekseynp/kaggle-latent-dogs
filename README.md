Setup:

- `conda env create -f conda.yaml`
- `conda activate kaggle-latent-dogs`
- `mkdir frames`

Single Breed:

- `rm frames/*`
- `python make_single_breed_video.py`
- `ffmpeg -i frames/%d.png -framerate 24 single.mp4`


Breed Grid:
- `rm frames/*`
- `python make_breed_grid_video.py`
- `ffmpeg -i frames/%d.png -framerate 24 many.mp4`
