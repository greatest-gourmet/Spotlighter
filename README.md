---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: label
    dtype:
      class_label:
        names:
          '0': abyssinian
          '1': american_bulldog
          '2': american_pit_bull_terrier
          '3': basset_hound
          '4': beagle
          '5': bengal
          '6': birman
          '7': bombay
          '8': boxer
          '9': british_shorthair
          '10': chihuahua
          '11': egyptian_mau
          '12': english_cocker_spaniel
          '13': english_setter
          '14': german_shorthaired
          '15': great_pyrenees
          '16': havanese
          '17': japanese_chin
          '18': keeshond
          '19': leonberger
          '20': maine_coon
          '21': miniature_pinscher
          '22': newfoundland
          '23': persian
          '24': pomeranian
          '25': pug
          '26': ragdoll
          '27': russian_blue
          '28': saint_bernard
          '29': samoyed
          '30': scottish_terrier
          '31': shiba_inu
          '32': siamese
          '33': sphynx
          '34': staffordshire_bull_terrier
          '35': wheaten_terrier
          '36': yorkshire_terrier
  - name: image_id
    dtype: string
  - name: label_cat_dog
    dtype:
      class_label:
        names:
          '0': cat
          '1': dog
  splits:
  - name: train
    num_bytes: 376746044.08
    num_examples: 3680
  - name: test
    num_bytes: 426902517.206
    num_examples: 3669
  download_size: 790265316
  dataset_size: 803648561.286
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
license: cc-by-sa-4.0
size_categories:
- 1K<n<10K
task_categories:
- image-classification
---

# The Oxford-IIIT Pet Dataset

## Description
A 37 category pet dataset with roughly 200 images for each class. The images have a large variations in scale, pose and lighting.

This instance of the dataset uses standard label ordering and includes the standard train/test splits. Trimaps and bbox are not included, but there is an `image_id` field that can be used to reference those annotations from official metadata.

Website: https://www.robots.ox.ac.uk/~vgg/data/pets/

## Citation
```bibtex
@InProceedings{parkhi12a,
  author       = "Omkar M. Parkhi and Andrea Vedaldi and Andrew Zisserman and C. V. Jawahar",
  title        = "Cats and Dogs",
  booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
  year         = "2012",
}
```