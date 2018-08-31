# chainer-for-cnn


## Directory Structure

CrownBread_Prediction/  
　　　　　　　　　├ model/  
　　　　　　　　　├ result/  
　　　　　　　　　├ model.py  
　　　　　　　　　├ data.py  
　　　　　　　　　├ train.py  
　　　　　　　　　└ predict.py

　

## Usage

### Training

    $ python train.py

Please set images like the following...

/path/to/img/  
　　　　├{ClassName1}/  
　　　　├{ClassName2}/  
　　　　├{ClassName3}/  
　　　　...  
　　　　└{ClassNameN}/  

The following commands are required.

    $ python train.py -i /path/to/img -n model -c 2 


-i : Path to images used for training  
-c : Number of image classed  
-n : The name of model used for training  

If you need more information, see the help.  

### Prediction

    $ python predict.py

Please set images like the following...  

/path/to/img/  
　　　　├img\_001.jpg  
　　　　├img\_002.jpg  
　　　　├img\_003.jpg  
　　　　...  
　　　　└img\_NNN.jpg  

The following commands are required.

    $ python predict.py -i /path/to/img -m /path/to/model -n NIN -c 2

-i : Path to images used for prediction  
-m : Path to model  
-c : Number of image classed  
-n : The name of model used for prediction  　

If you need more information, see the help.

## Requirement

    $ pip3 install argparse numpy cupy chainer matplotlib pillow tensorflow tensorflow-gpu

　

## Installation

    $ git clone https://github.com/yukitaka13-1110/chainer-for-cnn.git

