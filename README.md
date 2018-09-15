# Multi-Task
Multi Task and Multi Domain Learning Experiments with Neural Networks

# Abstract

> Neural networks are universal function approximators. In neural network parameter space there can exists a subset of region where parameters for predicting different multi task problem can also exists simultaneously. In a deep neural network if θi represents the parameter space for image generation or classification and θs for sequence. Then there exists θis where both parameters of sequence and image classification or generation can occur together.Used infoGAN's architecture but it should easily work with other types of conditional GAN's. The difference is in "conditional"-GAN we pass noise Z and class C in the generator inputs (Z, C) whereas in my experiment instead of class its tasks T (Z, T). That said theoratically we can also build a model with class for specific tasks by passing inputs as (Z, T, C).

Read more from here https://gananath.github.io/multi_task.html

# Requirments
- Python 2
- Keras
- TensorFlow
- scikit-image
- Matplotlib
- Numpy
- Pandas

# Citation
```
@misc{gananath2018,
  author = {Gananath, R.},
  title = {Multi-Task},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Gananath/Multi-Task}}
}
```
**Current Best after 3000 epochs of training**
![current_best](https://github.com/Gananath/gananath.github.io/blob/master/images/new_multi_pred.jpg)
