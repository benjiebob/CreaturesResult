# Creatures Great and SMAL - Result Viewer
Scripts for viewing Creatures Great and SMAL results.

## Installation
1. Clone the repository
   ```
   git clone https://github.com/benjiebob/CreaturesResult.git
    
2. Download texture map (from smal/dog_texture.pkl) and a version of SMAL 2017 converted to NumPy (smal_CVPR2017_np.pkl) from [my Google Drive](https://drive.google.com/open?id=1gPwA_tl1qrKiUkveE8PTsEOEMHtTw8br) and place under the smal folder

3. Download the preliminary [Maggie result data](https://drive.google.com/drive/folders/1dDx1Kncmd4W9wdKZaSBoUy8oHu2Hl5PI?usp=sharing)

4. Install dependencies, particularly [PyTorch (with cuda support)](https://pytorch.org/) and [PyTorch Port of Neural Mesh Renderer](https://github.com/daniilidis-group/neural_renderer)

5. Edit the IMAGE_DIR and RESULT_DIR paths in result_viewer.py to reflect your own file structure

6. Test the python3 script
   ```
   python result_viewer.py
   ```
