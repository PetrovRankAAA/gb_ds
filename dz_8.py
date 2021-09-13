import numpy as np
from numpy import mean, std, array
from scipy import stats

def ssd(list_, mean_):
    result = 0
    for el in list_:
        result += (el - mean_) ** 2
    return result


def ssd_factors(*lists):
    result = 0
    general_ = []

    # средняя общая
    for list_ in lists:
        general_ += list_
    mean_total = np.mean(general_)

    for list_ in lists:
        result += (np.mean(list_) - mean_total) ** 2 * len(list_)

    return result


def ssd_residuals(*lists):
    result = 0
    general_ = []

    for list_ in lists:
        mean_ = np.mean(list_)
        for el in list_:
            result += (el - mean_) ** 2

    return result


"""
Провести дисперсионный анализ для определения того, есть ли различия среднего роста среди взрослых футболистов,
хоккеистов и штангистов. Даны значения роста в трех группах случайно выбранных спортсменов: Футболисты: 173, 175,
180, 178, 177, 185, 183, 182. Хоккеисты: 177, 179, 180, 188, 177, 172, 171, 184, 180. Штангисты: 172, 173, 169, 177,
166, 180, 178, 177, 172, 166, 170.
"""
football = [173, 175, 180, 178, 177, 185, 183, 182]
hockey = [177, 179, 180, 188, 177, 172, 171, 184, 180]
lifters = [172, 173, 169, 177, 166, 180, 178, 177, 172, 166, 170]

general = football + hockey + lifters

k = len([lifters, football, hockey])
n = len(general)
alpha = 0.05
#Тестируем нулевую гипотезу
# H0 = mean_f = mean_h = mean_l
mean_f = np.mean(football)
mean_h = np.mean(hockey)
mean_l = np.mean(lifters)
mean_tl = np.mean(general)

# найти сумму квадратов отклонений общую
ssd_tl = ssd(general, mean_tl)
# и по группам
ssd_f = ssd_factors(football, hockey, lifters)

# факторная дисперсия - factor variance?
factor_var = ssd_f / (k - 1)

# сумма отклонений остатков
ssd_r = ssd_residuals(football, hockey, lifters)
# проверка значений
# print(ssd_r + ssd_f, ssd_tl) => 830.9642857142859 830.9642857142854

# остаточная дисперсия
residual_var = ssd_r / (n - k)

# критерий Фишера => 5.500053450812599
fischer_cr = factor_var / residual_var
# f_k = k - 1 = 2, f_n = n - k = 28 - 3 = 25
# Критерий Фишера для f_k = 2 и f_n = 25  -  3,38
# 5.5 < 3,38 - значения статистически значимые

check = stats.f_oneway(football, hockey, lifters)
# print(check) => F_onewayResult(statistic=5.500053450812596, pvalue=0.010482206918698694)
