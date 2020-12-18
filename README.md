## YoLo-V3 Knowledge Distillation

Original code from: https://github.com/ultralytics/yolov3 


This repository performs knowledge distillation between two yolo-v3 models: pre-trained teacher and student initialized from scratch using proxy datasets. 

### Environment

Install python 3.8 environment with following packages:

```
$ pip install -r requirements.txt
```

or use provided Dockerfile to create an image. 


### How to run?

1. Get access to `diode_yolo` directory as in top level repository. 
2. Extract a proxy dataset from `diode_yolo` directory to `/tmp` as follows:
   ``` 
   $ tar xzf /path/to/diode_yolo/hallucinate/hallucinate_320_normed.tgz -C /tmp
   ```
3. Extract coco dataset from `diode_yolo` directory to `/tmp` as follows: (for evaluation during training)
   ```
   $ tar xzf /path/to/diode_yolo/coco/coco.tgz -C /tmp
   ```
3. Copy yolo-v3 teacher weights file from `diode_yolo` to `weights` directory.
   ```
   cp /path/to/diode_yolo/pretrained/yolov3-spp-ultralytics.pt /path/to/lpr_deep_inversion/yolov3/weights/
   ```
3. Perform knowledge distillation on proxy dataset as follows:
   ```
   python distill.py --data NGC_hallucinate.data --weights '' --batch-size 64 --cfg yolov3-spp.cfg --device='0,1,2,3' --nw=20 --cfg-teacher yolov3-spp.cfg --weights-teacher './weights/yolov3-spp-ultralytics.pt' --alpha-yolo=0.0 --alpha-distill=1.0 --distill-method='mse'
   ```

Distillation and training logs are available at `diode_yolo/logs/` 

Available proxy dataset and their corresponding locations and `--data` flag for `distill.py` :

```

# Real/Rendered proxy datasets
coco  /path/to/diode_yolo/coco/coco.tgz  --data NGC_coco2014.data
GTA5  /path/to/diode_yolo/gta5/gta5.tgz  --data NGC_gta5.data
bdd100k  /path/to/diode_yolo/bdd100k/bdd100k.tar.gz  --data NGC_bdd100k.data
voc  /path/to/diode_yolo/voc/voc.tgz  --data NGC_voc.data
imagenet  /path/to/diode_yolo/imagenet/imagenet.tgz  --data NGC_imagenet.data

# DIODE generated proxy datasets
diode-coco  /path/to/diode_yolo/fakecoco/fakecocov3.tgz  --data NGC_fakecoco.data
diode-onebox  /path/to/diode_yolo/onebox/onebox.tgz  --data NGC_onebox.data
diode-onebox w/ fp sampling  /path/to/diode_yolo/hallucinate/hallucinate_320_normed.tgz  --data NGC_hallucinate.data
diode-onebox w/ tiles  /path/to/diode_yolo/onebox_tiles_coco/tiles.tgz  --data NGC_tiles.data
```

Modify `./run.sh` to choose which proxy dataset for knowledge distillation.  

