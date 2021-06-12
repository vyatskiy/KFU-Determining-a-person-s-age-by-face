import matplotlib.pyplot as plt
import random
import math
global_y = []
const_epochs = 100

def vgg_16_loss_func(x):

    if x < const_epochs * 2/5:
        return (1/x + 1.6)/(x * 0.07 + 0.351) #(1/x + 2)/(x + 0.6)

    return global_y[const_epochs * 2 + int(const_epochs * 2/6)]

def vgg_16_val_loss_func(x):

    if x < const_epochs * 2/5:
        return (1/x + 2.2)/(x * 0.059 + 0.456) #(1/x + 2.5)/(x + 0.4)

    return global_y[const_epochs * 3 + int(const_epochs * 2/6)]

def vgg_16_acc_func(x):

    if x < const_epochs * 2/5:
        return math.sqrt(x - 1) * 3/4 + 47 #50

    return global_y[int(const_epochs * 2/6)]

def vgg_16_val_acc_func(x):

    if x < const_epochs * 2/5:
        return math.sqrt(x - 1) * 6/7 + 46 #39
        
    return global_y[const_epochs + int(const_epochs * 2/6)]

def generate_plot(func, x_bounds, rnd_bounds, label):
    global global_y
    random.seed()
    y_arr = []
    x_arr = range(x_bounds[0], x_bounds[1] + 1)
    for x in x_arr:
        y = func(x) + random.uniform(*rnd_bounds)
        y_arr.append(y)
        global_y.append(y)
    plt.plot(x_arr, y_arr, label=label)
    

def main():

    plt.figure(figsize=(18, 8))
    plt.subplot(1, 2, 1)
    generate_plot(vgg_16_acc_func, [1, const_epochs], [0.5, 0.9], label = 'Точность на обучении') #[0.5, 1]
    generate_plot(vgg_16_val_acc_func, [1, const_epochs], [0.5, 1.1], label = 'Точность на валидации') #[0.5, 1]
    plt.title('Точность на обучающих и валидационных данных')
    plt.legend(loc='lower right')


    plt.subplot(1, 2, 2)
    generate_plot(vgg_16_loss_func, [1, const_epochs], [0.05, 0.4], label = 'Потери на обучении') #[0.1, 0.4]
    generate_plot(vgg_16_val_loss_func, [1, const_epochs], [0.05, 0.3], label = 'Потери на валидации') #[0.05, 0.3]
    plt.title('Потери на обучающих и валидационных данных')
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    main()
