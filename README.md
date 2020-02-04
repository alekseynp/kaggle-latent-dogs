Setup:

- `conda env create -f conda.yaml`
- `conda activate kaggle-latent-dogs`

Single Breed:

- `mkdir single_frames`
- `rm single_frames/*`
- `python make_single_breed_video.py`
- `ffmpeg -i single_frames/%d.png -framerate 24 single.mp4 -preset veryslow -crf 0 -tune film`


Breed Grid:

- `mkdir breed_frames`
- `rm breed_frames/*`
- `python make_breed_grid_video.py`
- `ffmpeg -i breed_frames/%d.png -framerate 24 many.mp4 -preset veryslow -crf 0 -tune film`

