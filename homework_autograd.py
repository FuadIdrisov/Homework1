import torch
### 2.1 Простые вычисления с градиентами (8 баллов)
x = torch.tensor([1.0], requires_grad=True)
y = torch.tensor([2.0], requires_grad=True)
z = torch.tensor([3.0], requires_grad=True)

f = x**2 + y**2 + z**2 + 2 * x* y * z

#Вычисляем градиенты
f.backward()

#Нахождения градиента x,y,z
grad_x_tensor = x.grad.item()
grad_y_tensor = y.grad.item()
grad_z_tensor = z.grad.item()

print(f"2.1 Простые вычисления с градиентами:\nВывод функции f:\n{f}")
print(f"Градиент тензора X:\n{grad_x_tensor}")
print(f"Градиент тензора Y:\n{grad_y_tensor}")
print(f"Градиент тензора Z:\n{grad_z_tensor}")

#Аналитическая проверка
df_dx = 2*x + 2*y*z
df_dy = 2*y + 2*x*z
df_dz = 2*z + 2*x*y

print("\nАналитические градиенты:")
print(f"df/dx = {df_dx.item()}")
print(f"df/dy = {df_dy.item()}")
print(f"df/dz = {df_dz.item()}")

# Проверка совпадения
assert torch.allclose(x.grad, df_dx)
assert torch.allclose(y.grad, df_dy)
assert torch.allclose(z.grad, df_dz)

### 2.2 Градиент функции потерь (9 баллов)
"""Вычисляет MSE и её градиенты по параметрам w и b.
    
    Параметры:
    x : array-like, список признаков (n элементов)
    y_true : array-like, список истинных значений (n элементов)
    w : float, вес модели
    b : float, смещение модели
    
    Возвращает:
    mse : float, значение MSE
    grad_w : float, градиент MSE по w
    grad_b : float, градиент MSE по b"""

def mse_gradients(x, y_true, w, b):
    n = len(x)
    total_error = 0.0
    grad_w_sum = 0.0
    grad_b_sum = 0.0
    
    # Проходим по всем элементам выборки
    for i in range(n):
        # Предсказанное значение и ошибка для текущего элемента
        y_pred_i = w * x[i] + b      
        error = y_pred_i - y_true[i]  
        
        # Суммируем квадраты ошибок
        total_error += error ** 2     
        
        # Суммы для градиентов
        grad_w_sum += error * x[i]
        grad_b_sum += error
    
    # Вычисляем MSE и градиенты
    mse = total_error / n
    grad_w = (2 / n) * grad_w_sum
    grad_b = (2 / n) * grad_b_sum
    
    return mse, grad_w, grad_b

x = [1, 2, 3, 4]
y_true = [2, 4, 6, 8]
w = 1.5  # Пример веса
b = 0.5  # Пример смещения

# Расчёт MSE и градиентов
mse, grad_w, grad_b = mse_gradients(x, y_true, w, b)
print(f"\n2.2 Градиент функции потерь\nMSE: {mse:.4f}")
print(f"Градиент w: {grad_w}")
print(f"Градиент b: {grad_b}") 

### 2.3 Цепное правило (8 баллов)
def analytical_gradient(x):
    """Аналитическое вычисление градиента"""
    return 2 * x * torch.cos(x**2 + 1)

def check_gradient(x_val):
    """Сравнение аналитического и автоматического градиента"""
    # Создаем тензор с отслеживанием градиентов
    x = torch.tensor([x_val], dtype=torch.float32, requires_grad=True)
    
    # Вычисляем функцию
    f = torch.sin(x**2 + 1)
    
    # Вычисляем градиент автоматически
    auto_grad = torch.autograd.grad(f, x, create_graph=True)[0]
    
    # Вычисляем аналитический градиент
    anal_grad = analytical_gradient(x)
    
    # Сравниваем результаты
    print(f"\nДля x = {x_val:.2f}:")
    print(f"Автоматический градиент: {auto_grad.item():.6f}")
    print(f"Аналитический градиент: {anal_grad.item():.6f}")
    print(f"Разница: {torch.abs(auto_grad - anal_grad).item():.6e}")

# Проверка для разных значений x
test_values = [0.5, 1.0, 1.5, 2.0, 3.0]
for val in test_values:
    check_gradient(val)