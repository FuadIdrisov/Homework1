import torch

### 1.1 Создание тензоров (7 баллов)
random_tensor = torch.rand(3, 4)

zeros_tensor = torch.zeros(2, 3, 4)

ones_tensor = torch.ones(5, 5)

reshape_tensor = torch.arange(0, 16).reshape(4, 4)
print(f"1.1 Создание тензоров:\nТензор размером 3x4, заполненный случайными числами от 0 до 1\n{random_tensor}")
print(f"Тензор размером 2x3x4, заполненный нулями:\n{zeros_tensor}")
print(f"Тензор размером 5x5, заполненный единицами:\n{ones_tensor}")
print(f"Тензор размером 4x4 с числами от 0 до 15 :\n{reshape_tensor}\n")

### 1.2 Операции с тензорами (6 баллов)
A_tensor = torch.rand(3, 4)

B_tensor = torch.rand(4, 3)

transposed_tensor = A_tensor.T
transposed_tensor_B = B_tensor.T

matmul_tenzor = torch.matmul(A_tensor, B_tensor )

mul_tensor = torch.mul(A_tensor, transposed_tensor_B)

sum_tensor = torch.sum(A_tensor)
print(f"1.2 Операции с тензорами:\nТранспонирование тензора A:\n{transposed_tensor}")
print(f"Матричное умножение A и B:\n{matmul_tenzor}")
print(f"Поэлементное умножение A и транспонированного B:\n{mul_tensor}")
print(f"Cумма всех элементов тензора A:\n{sum_tensor}\n")

### 1.3 Индексация и срезы (6 баллов)
# Создаём тензор размером 5x5x5 с числами от 0 до 9
create_tensor = torch.arange(125).reshape(5, 5, 5)

first_row = create_tensor[:, 0 , :]

last_columns = create_tensor[:, :, -1]

central_tensor = create_tensor[:, 1:4:2, 1:4:2]

index_tensor = create_tensor[::2, ::2, ::2]
print(f"1.3 Индексация и срезы:\nИзвлечение первой строки:\n{first_row}")
print(f"Извлечение последнего столбца:\n{last_columns}")
print(f"Извлечение подматрицы размером 2x2 из центра тензора:\n{central_tensor}")
print(f"Извлекаем элементы с четными индексами:\n{index_tensor}")

### 1.4 Работа с формами (6 баллов)
tensor = torch.arange(24)

tensor_2x12 = torch.reshape(tensor, (2, 12))

tensor_3x8 = torch.reshape(tensor, (3, 8))

tensor_4x6 = torch.reshape(tensor, (4, 6))

tensor_2x3x4 = torch.reshape(tensor, (2, 3, 4))

tensor_2x2x2x3 = torch.reshape(tensor, (2, 2, 2, 3))
print(f"1.4 Работа с формами:\nМеняем форму на 2x12:\n{tensor_2x12}")
print(f"Меняем форму на 3x8:\n{tensor_3x8}")
print(f"Меняем форму на 4x6:\n{tensor_4x6}")
print(f"Меняем форму на 2x3x4:\n{tensor_2x3x4}")
print(f"Меняем форму на 2x2x2x3:\n{tensor_2x2x2x3}")