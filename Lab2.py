import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def solve_lab_exact(intervals, frequencies, gamma, title_prefix=""):
    n = sum(frequencies)
    midpoints = [(a + b) / 2.0 for a, b in intervals]
    
    print("\n" + "="*75)
    print(f"   ТОЧНІ РЕЗУЛЬТАТИ РОЗРАХУНКІВ ({title_prefix})")
    print("="*75)
    
    # --- ЗАВДАННЯ 1 ---
    print("\n--- ЗАВДАННЯ 1: Перевірка та дискретний ряд ---")
    print(f"Сума частот sum(n_i) = {n}")
    print(f"Заданий обсяг вибірки n = {n}")
    print("Дискретний варіаційний ряд:")
    print(f"{'№':<3} | {'Інтервал':<10} | {'Середина (x_i)':<15} | {'Частота (n_i)':<12} | {'Відносна частота (w_i)':<20}")
    print("-" * 72)
    for i, ((a, b), x_i, n_i) in enumerate(zip(intervals, midpoints, frequencies), 1):
        w_i = n_i / n
        print(f"{i:<3} | {f'{a}-{b}':<10} | {x_i:<15.1f} | {n_i:<12} | {w_i:<20.4f}")
        
    # --- ЗАВДАННЯ 3 ---
    sum_xi_ni = sum(x * n_i for x, n_i in zip(midpoints, frequencies))
    x_mean = round(sum_xi_ni / n, 4)
    
    sum_dev_sq = 0.0
    for x_i, n_i in zip(midpoints, frequencies):
        dev = round(x_i - x_mean, 4)
        dev_sq = round(dev ** 2, 4)
        sum_dev_sq += round(n_i * dev_sq, 4)
        
    sum_dev_sq = round(sum_dev_sq, 4)
    var_sample = round(sum_dev_sq / n, 4)
    var_corrected = round((n / (n - 1)) * var_sample, 4)
    s_corrected = round(np.sqrt(var_corrected), 4)
    
    print("\n--- ЗАВДАННЯ 3: Точкові оцінки ---")
    print(f"1. Вибіркове середнє (x_mean)                = {x_mean:.4f}")
    print(f"2. Вибіркова дисперсія (d_bar)                = {var_sample:.4f}")
    print(f"3. Виправлена дисперсія (s^2)                 = {var_corrected:.4f}")
    print(f"4. Виправлене середньоквадратичне відх. (s)   = {s_corrected:.4f}")
    
    # --- ЗАВДАННЯ 4 ---
    df = n - 1
    t_crit = round(stats.t.ppf((1 + gamma) / 2, df), 4)
    delta = round(t_crit * (s_corrected / np.sqrt(n)), 4)
    ci_m_low = round(x_mean - delta, 4)
    ci_m_high = round(x_mean + delta, 4)
    
    print("\n--- ЗАВДАННЯ 4: Довірчий інтервал для m (Стьюдент) ---")
    print(f"Критичне значення Стьюдента t({gamma}, df={df}) = {t_crit:.4f}")
    print(f"Точність оцінки (delta)                        = {delta:.4f}")
    print(f"Довірчий інтервал для m                        = ({ci_m_low:.4f}; {ci_m_high:.4f})")
    
    # --- ЗАВДАННЯ 5 ---
    alpha = 1 - gamma
    chi2_low = round(stats.chi2.ppf(alpha / 2, df), 2)
    chi2_high = round(stats.chi2.ppf(1 - alpha / 2, df), 2)
    
    ci_var_low = round((df * var_corrected) / chi2_high, 4)
    ci_var_high = round((df * var_corrected) / chi2_low, 4)
    
    ci_std_low = round(np.sqrt(ci_var_low), 4)
    ci_std_high = round(np.sqrt(ci_var_high), 4)
    
    print("\n--- ЗАВДАННЯ 5: Довірчі інтервали для дисперсії та СКВ ---")
    print(f"Критичні значення Хі-квадрат: chi2_0.025 = {chi2_low:.2f}, chi2_0.975 = {chi2_high:.2f}")
    print(f"Довірчий інтервал для дисперсії (sigma^2)      = ({ci_var_low:.4f}; {ci_var_high:.4f})")
    print(f"Довірчий інтервал для СКВ (sigma)             = ({ci_std_low:.4f}; {ci_std_high:.4f})")
    
    # --- ЗАВДАННЯ 6 ---
    rel_accuracy = round((delta / x_mean) * 100, 2)
    width_m = round(ci_m_high - ci_m_low, 4)
    width_var = round(ci_var_high - ci_var_low, 4)
    
    print("\n--- ЗАВДАННЯ 6: Відносна точність та ширина інтервалів ---")
    print(f"Відносна точність оцінки середнього (epsilon)  = {rel_accuracy:.2f}%")
    print(f"Ширина інтервалу для m                         = {width_m:.4f}")
    print(f"Ширина інтервалу для sigma^2                   = {width_var:.4f}")
    print("Пояснення несиметричності для sigma^2:")
    print("   Розподіл Хі-квадрат є асиметричним, тому критичні точки chi2_left")
    print("   та chi2_right розташовані несиметрично відносно s^2.")
    
    # --- ЗАВДАННЯ 7 (ОНОВЛЕНЕ ПІД ТАБЛИЦЮ 1.3) ---
    gammas = [0.90, 0.95, 0.99]
    print("\n--- ЗАВДАННЯ 7: Таблиця 1.3 — Вплив надійності на довірчі інтервали ---")
    print(f"{'gamma':<8} | {'t_gamma':<8} | {'delta':<8} | {'Довірчий інтервал для m':<26} | {'Довірчий інтервал для sigma^2':<28}")
    print("-" * 88)
    
    for g in gammas:
        t_val = round(stats.t.ppf((1 + g) / 2, df), 4)
        d_val = round(t_val * (s_corrected / np.sqrt(n)), 4)
        ci_m_str = f"({round(x_mean - d_val, 4):.4f}; {round(x_mean + d_val, 4):.4f})"
        
        a_val = 1 - g
        chi2_l = stats.chi2.ppf(a_val / 2, df)
        chi2_h = stats.chi2.ppf(1 - a_val / 2, df)
        ci_v_low = round((df * var_corrected) / chi2_h, 4)
        ci_v_high = round((df * var_corrected) / chi2_l, 4)
        ci_v_str = f"({ci_v_low:.4f}; {ci_v_high:.4f})"
        
        print(f"{g:<8.2f} | {t_val:<8.4f} | {d_val:<8.4f} | {ci_m_str:<26} | {ci_v_str:<28}")
        
    # --- ЗАВДАННЯ 8 ---
    n_required = 4 * n
    print("\n--- ЗАВДАННЯ 8: Обсяг вибірки для покращення точності вдвічі ---")
    print(f"Щоб точність delta покращилася вдвічі (зменшилася в 2 рази),")
    print(f"необхідний обсяг вибірки n' = 4 * n = 4 * {n} = {n_required}")
    print("="*75 + "\n")
    
    # --- ЗАВДАННЯ 2 ---
    print("Ввімкнено Завдання 2: Відображення полігону та гістограми частот...")
    bin_edges = [intervals[0][0]] + [b for a, b in intervals]
    
    plt.figure(figsize=(10, 6))
    plt.hist(midpoints, weights=frequencies, bins=bin_edges, edgecolor='black', alpha=0.6, color='skyblue', label='Гістограма частот')
    plt.plot(midpoints, frequencies, marker='o', color='red', linewidth=2, linestyle='-', label='Полігон частот')
    
    plt.title(f'Завдання 2: Гістограма та полігон частот ({title_prefix})', fontsize=14)
    plt.xlabel('Значення ознаки (x_i)', fontsize=12)
    plt.ylabel('Частота (n_i)', fontsize=12)
    plt.xticks(bin_edges)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

def run_variant_9():
    intervals = [(24, 26), (26, 28), (28, 30), (30, 32), (32, 34), (34, 36), (36, 38), (38, 40)]
    frequencies = [2, 5, 24, 23, 28, 10, 3, 4]
    gamma = 0.95
    solve_lab_exact(intervals, frequencies, gamma, title_prefix="Варіант 9")

def run_custom_variant():
    print("\n--- Введення власних даних ---")
    try:
        k = int(input("Введіть кількість інтервалів: "))
        intervals = []
        frequencies = []
        
        print("Введіть межі та частоту для кожного інтервалу:")
        for i in range(k):
            print(f"Інтервал {i+1}:")
            a = float(input("  Нижня межа: "))
            b = float(input("  Верхня межа: "))
            n_i = int(input("  Частота n_i: "))
            intervals.append((a, b))
            frequencies.append(n_i)
            
        gamma = float(input("Введіть надійність gamma (наприклад, 0.95): "))
        solve_lab_exact(intervals, frequencies, gamma, title_prefix="Власні дані")
    except Exception as e:
        print(f"\nПомилка введення даних: {e}. Будь ласка, спробуйте ще раз.")

def main_menu():
    while True:
        print("\n" + "="*35)
        print("        ГОЛОВНЕ МЕНЮ")
        print("="*35)
        print("1. Мій варіант (Варіант 9)")
        print("2. Ввести свої значення")
        print("3. Вийти")
        print("="*35)
        
        choice = input("Оберіть пункт меню (1-3): ").strip()
        
        if choice == '1':
            run_variant_9()
        elif choice == '2':
            run_custom_variant()
        elif choice == '3':
            print("\nПрограму завершено. Успіхів у навчанні!")
            break
        else:
            print("\nНекоректний вибір. Введіть 1, 2 або 3.")

if __name__ == "__main__":
    main_menu()