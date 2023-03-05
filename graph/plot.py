import numpy as np
import matplotlib.pyplot as plt


with open("settings.txt", "r") as settings:
    tmp = [float(i) for i in settings.read().split("\n")]
dt = tmp[1]

data_array = np.loadtxt("data.txt", dtype=float)
time = [20,21,22,23,24,25, 
        20,21,22,23,24,25]#np.linspace(0, dt * (len(data_array) - 1), len(data_array))

fig, ax = plt.subplots(figsize=(16, 10), dpi=400)
ax.set_title("Сравнение скорости std::sort() (i5-10300H) и bsort (RTX 3060 mobile) [type = float]")
ax.set_ylabel("Время, t")
ax.set_xlabel("2^N, эл-в")

ax.plot(time[0:6], data_array[0:6], label='bsort()', color='b', linewidth=3) 
ax.plot(time[6:13], data_array[6:13], label='std::sort()', color='r', linewidth=3) 
ax.legend(fontsize = 10,
          ncol = 1,    #  количество столбцов
          facecolor = 'oldlace',    #  цвет области
          edgecolor = 'r',    #  цвет крайней линии
          title = 'Данные',    #  заголовок
          title_fontsize = '10'    #  размер шрифта заголовка
         )
ax.grid(which='major',
        color = 'k')

ax.minorticks_on()

ax.grid(which='minor',
        color = 'gray',
        linestyle = ':')


yMax = np.max(data_array)
ax.set_xlim([19, 26])
ax.set_ylim([0, yMax + 1])

vMax = np.argmax(data_array)
t1 = time[vMax]
t2 = time[-1] - t1


#plt.show()
fig.savefig("save.svg")

#print(data_array)