# імпортуємо бібліотеки TensorFlow для роботи з нейронними мережами, Matplotlib для візуалізації даних та NumPy для роботи з масивами чисел
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# завантажує набір даних MNIST який містить рукописні цифри від 0 до 9 разом з відповідними мітками
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# значення пікселів перетворюються з діапазону від 0 до 255 до діапазону від 0 до 1 для нормалізації даних
x_train, x_test = x_train / 255.0, x_test / 255.0

# розширюємо розмірність даних додавши одну розмірність для каналу зображення (RGB).
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# будуємо CNN модель
model = models.Sequential()


# додаємо згортковий шар з 32 фільтрами розміром (3, 3) та активацією ReLU
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# додаємо шар пулінгу який зменшує розмірність зображення за допомогою максимального зведення на підматрицях розміром (2, 2)
model.add(layers.MaxPooling2D((2, 2)))

# додаємо ще один згортковий шар з 64 фільтрами розміром (3, 3) та активацією ReLU
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# додоаємо ще шар пулінгу та згортковий шар для витягнутих ознак
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# гладжуємо вихід перетворюючи його в одномірний вектор перед передачею його в повністю зв'язаний шар
model.add(layers.Flatten())

# додаваємо повністю зв'язаний шар з 64 нейронами та активацією ReLU
model.add(layers.Dense(64, activation='relu'))

# dense шар. 10 нейронів, активація softmax
model.add(layers.Dense(10, activation='softmax'))

# Компілюємо модель визначаючи оптимізатор функцію втрати та метрики
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Тренуєм модель протягом 5 епох
model.fit(x_train, y_train, epochs=5)

# Оцінюємо точність
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

# Робимо прогнози на тестовому наборі
predictions = model.predict(x_test)


# Візуалізуємо прогнози
num_rows = 5
num_cols = 5
plt.figure(figsize=(12, 12))

for i in range(num_rows * num_cols):
    plt.subplot(num_rows, num_cols, i + 1)
    plt.imshow(x_test[i, :, :, 0], cmap='gray')
    true_label = y_test[i]

    # Показати прогнозовану цифри та її ймовірність
    predicted_label = np.argmax(predictions[i])
    confidence = np.max(predictions[i])

    plt.title(f'Цифра: {true_label}\nПрогнозована цифра: {predicted_label}\nЙмовірність: {(confidence * 100):.1f}%')

    plt.axis('off')

plt.tight_layout()
plt.show()