# mos2_segmentation
## Usage
` python train.py -tr [path to traning data] -ta [path to target data] -e [epoch数] -b [バッチサイズ] -o [path to output] -m [model number] -w [0 or 1] -bi [0 or 1]`

- model
  - 0 : fcn32s
  - 1 : unet
  - 2 : unet2
  - 3 : pix2pix
## Sample
' python train.py -e 100 -o ./result/ -ta ./data/label_012/ -g 0 -m 2 -w 0 `
