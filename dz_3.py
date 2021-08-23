from math import sqrt
import numpy as np


def my_mean(array):
    return sum(array) / len(array)


def my_std_deviation(array, uncorrected = True):
    mean = my_mean(array)
    n = len(array)
    mu = 0

    for i in range(n):
        mu += (array[i] - mean) ** 2

    if uncorrected:
        return mu / n
    elif not uncorrected:
        return mu / (n - 1)


def combination(n, k):
    result = 1
    n_minus_k = n - k
    # n! / k!
    while n > k:
        result *= n
        n -= 1
    # (n! / k!) / (n-k)!
    while n_minus_k > 1:
        result /= n_minus_k
        n_minus_k -= 1

    return result


"""
1. Даны значения зарплат из выборки выпускников: 100, 80, 75, 77, 89, 33, 45, 25, 65, 17, 30, 24, 57, 55, 70, 75,
65, 84, 90, 150. Посчитать (желательно без использования статистических методов наподобие std, var, mean) среднее
арифметическое, среднее квадратичное отклонение, смещенную и несмещенную оценки дисперсий для данной выборки.
"""
incomes = [ 100, 80, 75, 77, 89, 33, 45, 25, 65, 17, 30, 24, 57, 55, 70, 75,
65, 84, 90, 150 ]
# среднее арифметическое, 65.3
print(my_mean(incomes))
# смещенная оценка дисперсии, 950.11
print(my_std_deviation(incomes))
# несмещенная оценка дисперсии, 1000.1157894736842
print(my_std_deviation(incomes, uncorrected=False))
# среднее квадратичное отклонение, 30.823854398825596
print(sqrt(my_std_deviation(incomes)))

"""
2. В первом ящике находится 8 мячей, из которых 5 - белые. Во втором ящике - 12 мячей, из которых 5 белых. Из 
первого ящика вытаскивают случайным образом два мяча, из второго - 4. Какова вероятность того, что 3 мяча белые? 
"""
# существует 3 ситуации:
# 1. Два мяча(белые) из первого ящика, один из второй
# 2. Один из первого и два из второго
# 3. Все три из второго ящика

# Вероятность первой ситуации, 0.12626262626262627 ~ 12%
first_case = combination(5, 2) * combination(3, 0) / combination(8, 2) * combination(5, 1) * combination(7, 3) / combination(12, 4)
# Вероятность второй ситуации, 0.22727272727272724 ~ 22%
secnd_case = combination(5, 1) * combination(3, 1) / combination(8, 2) * combination(5, 2) * combination(7, 2) / combination(12, 4)
# Вероятность третьей ситуации, 0.01515151515151515 ~ 1%
third_case = combination(5, 0) * combination(3, 2) / combination(8, 2) * combination(5, 3) * combination(7, 1) / combination(12, 4)

# Общая вероятность равна 0.3686868686868686 ~ 36%
print(first_case + secnd_case + third_case)
"""
3.На соревновании по биатлону один из трех спортсменов стреляет и попадает в мишень. Вероятность попадания для 
первого спортсмена равна 0.9, для второго — 0.8, для третьего — 0.6. Найти вероятность того, что выстрел произведен: 
a). первым спортсменом б). вторым спортсменом в). третьим спортсменом 
"""
#не совсем понял задачу

"""
4.В университет на факультеты A и B поступило равное количество студентов, а на факультет C студентов поступило 
столько же, сколько на A и B вместе. Вероятность того, что студент факультета A сдаст первую сессию, равна 0.8. Для 
студента факультета B эта вероятность равна 0.7, а для студента факультета C - 0.9. Студент сдал первую сессию. 
Какова вероятность, что он учится: a). на факультете A б). на факультете B в). на факультете C? 
"""
# общая доля студентов сдавших сессию:
# 0.8 * 0.25 + 0.7 * 0.25 + 0.9 * 0.5 = 0.2 + 0.175 + 0.45 = 0.825
# а) 0.8 * 0.25 / 0.825 = 0.242 = 24.2%
# б) 0.7 * 0.25 / 0.825 = 0.212 = 21.2%
# в) 0.9 * 0.5 / 0.825 = 0.545 = 54.5%

"""
5.Устройство состоит из трех деталей. Для первой детали вероятность выйти из строя в первый месяц равна 0.1, 
для второй - 0.2, для третьей - 0.25. Какова вероятность того, что в первый месяц выйдут из строя: а). все детали б). 
только две детали в). хотя бы одна деталь г). от одной до двух деталей? 
"""
# а) 0.1 * 0.2 * 0.25 = 0.005 = 0.5%
# б) 0.9 * 0.2 * 0.25(первая деталь уцелела) + 0.1 * 0.8 * 0.25 + 0.1 * 0.2 * 0.75 = 0.045 + 0.02 + 0.015 = 0.08 = 8%
# в) 1 - 0.9 * 0.8 * 0.75 = 0.46 = 46%
# г) вероятность что две детали сломаются 8%
#    прибавить к этому вероятность поломок только одной детали
#    0.1 * 0.8 * 0.75(первая деталь сломалась) + 0.9 * 0.2 * 0.75 + 0.9 * 0.8 * 0.25 = 0.06 + 0.135 + 0.18 = 0.375 = 37.5%
#    37.5 + 8 = 45.5%