import tensorflow as tf

if tf.test.gpu_device_name():
      print(f'Default GPU Device:{tf.test.gpu_device_name()}')
else: print("Please check your installation")