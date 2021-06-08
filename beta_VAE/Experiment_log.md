# Model Structure:

## 1st Generation:
-  Latent Dimension: MNIST 8  , Celeb 64 
- Encoder: 2 Conv2D(layers 3 by 3 filter), decoder: 2 Conv2D (2,2) filter.
```python
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=shape),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=self.output_f * self.output_f *32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(self.output_f, self.output_f, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=self.output_s, kernel_size=3, strides=1, padding='same'),
        ]
    )

```

## 2nd Generation (3/17/21)
-  Latent Dimension: MNIST 8  , Celeb 64 
-  3 conv2d encoder, 3 conv2d layers decoder

## 5/17 model (s_decoder train to encode angle and apply the back propogate )
- MNIST test 7 full range digit 7, partial range digit 9;
- 5/21 mode test 8 full range digit 7, partial range digit 3; 
- 5/26 mode test9 full range digit 7, partial range digit 3 with training method(futher_dis) setting, ;
- 5/31: 
    - mnist_test 12 full range digit [4,5,6], partial range digit 3;
    - teacher_network full range digit 7,
- 6/6:
    - training data:
        -   full range: 7 [:100], parital range:9 [:100] 
    - teacher_network1, without bias vector (both decoder and encoder)
    - teacher_network2, without bias (decoder) 
    - student_network1, without bias (decoder)
    
- mnist_test14 (6/6):
    - beta_tcvae model
    - full range digit 4, partial range:3