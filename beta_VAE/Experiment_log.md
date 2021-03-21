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
