# run_petroleum_models.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from petroleum_generation import (
    SingleComponentGeneration,
    MultiComponentGeneration,
    SweeneyBurnhamModel,
    DuttaModel,
    SteraneAromatization_McKenzie1983
)

def run_single_component_example():
    """Пример запуска однокомпонентной генерации"""
    print("=== Однокомпонентная генерация ===")
    
    single_gen = SingleComponentGeneration()
    
    # Упрощенные параметры для теста
    A = 3.17 * 10**11 * 1000  # [s-1]
    S_i0 = [0.05, 0.09, 0.17, 0.27, 0.21]
    S_i0 = np.array(S_i0) / 100
    
    E_a = np.array([47, 48, 49, 50, 51])
    
    # Температурные интервалы
    T_intervals = [
        (0, 20), (20, 40), (40, 60), (60, 80)
    ]
    
    # Скорости нагрева
    HRs = np.array([1e-15] * len(T_intervals))
    
    try:
        # Расчет
        total_conv, temps, conversions, df = single_gen.calculate_conversion_non_isothermal(
            S_i0, A, E_a, T_intervals, HRs
        )
        
        print(f"Общая конверсия: {total_conv:.4f}")
        print(df)
        
        # Построение графика
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(conversions)), conversions, marker='o')
        plt.xlabel("Временной шаг")
        plt.ylabel("Конверсия")
        plt.title("Однокомпонентная генерация - эволюция конверсии")
        plt.grid(True)
        plt.show()
        
    except Exception as e:
        print(f"Ошибка в однокомпонентной генерации: {e}")

def run_multi_component_example():
    """Пример запуска многокомпонентной генерации"""
    print("\n=== Многокомпонентная генерация ===")
    
    multi_gen = MultiComponentGeneration()
    
    try:
        # Упрощенные параметры для 2 компонентов
        A = [3.17 * 10**11 * 10, 3.17 * 10**11 * 50]
        
        S_i0 = [
            [0.01, 0.03, 0.12, 0.37, 0.99],  # газ
            [0.01, 0.03, 0.19, 0.78, 2.53]   # нефть
        ]
        S_i0 = [np.array(i) / 100 for i in S_i0]
        
        E_a = [
            np.array([39, 40, 41, 42, 43]),  # газ
            np.array([40, 41, 42, 43, 44])   # нефть
        ]
        
        Ratio = [17*1e-2, 83*1e-2]
        
        # Температурные интервалы
        T_intervals = [
            (0, 20), (20, 40), (40, 60), (60, 80)
        ]
        
        HRs = np.array([1e-15] * len(T_intervals))
        
        # Расчет для каждого компонента
        TR_components = []
        for idx, (S_i0_j, E_a_j) in enumerate(zip(S_i0, E_a)):
            conversions = multi_gen.calculate_conversion_non_isothermal_multi_for_mass_gen(
                S_i0_j, A[idx], E_a_j, T_intervals, HRs, Ratio[idx]
            )
            TR_components.append(conversions)
        
        # Визуализация
        plt.figure(figsize=(10, 6))
        component_names = ['Газ', 'Нефть']
        colors = ['red', 'blue']
        
        for i, (conversions, name, color) in enumerate(zip(TR_components, component_names, colors)):
            plt.plot(range(len(conversions)), conversions, marker='o', color=color, label=name)
        
        plt.xlabel("Временной шаг")
        plt.ylabel("Конверсия")
        plt.title("Многокомпонентная генерация")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    except Exception as e:
        print(f"Ошибка в многокомпонентной генерации: {e}")

def run_vitrinite_reflectance_example():
    """Пример запуска модели витринита"""
    print("\n=== Модель отражательной способности витринита ===")
    
    try:
        # Создаем тестовые данные с правильным количеством интервалов
        data = {
            'Время, млн лет': [0, 10, 20, 30, 40]  # 5 точек = 4 интервала
        }
        df_temperature = pd.DataFrame(data)
        
        # Создаем температурные интервалы для разных слоев
        intervals = {}
        for i in range(1, 3):  # 2 слоя
            T_intervals = []
            # Количество температурных интервалов должно быть на 1 меньше, чем временных точек
            for j in range(len(df_temperature) - 1):
                T_intervals.append((j*20, (j+1)*20))
            intervals[f'T_intervals_{i}'] = T_intervals
        
        print(f"Количество временных точек: {len(df_temperature)}")
        print(f"Количество температурных интервалов в слое: {len(T_intervals)}")
        
        # Запуск модели
        model = SweeneyBurnhamModel()
        last_Ro_values, ro_layers = model.calculate_ro_values(intervals, df_temperature)
        
        print("Успешно рассчитаны значения Ro для слоев:")
        for i, value in enumerate(last_Ro_values):
            print(f"Слой {i+1}: Ro = {value:.4f}")
            
        # Построение графика
        model.plot_ro_vs_time(intervals, df_temperature)
        
    except Exception as e:
        print(f"Ошибка в модели витринита: {e}")
        import traceback
        traceback.print_exc()

def run_illite_conversion_example():
    """Пример запуска модели трансформации иллита"""
    print("\n=== Модель трансформации иллита ===")
    
    try:
        # Тестовые данные
        data = {
            'Время, млн лет': [0, 10, 20, 30]  # 4 точки = 3 интервала
        }
        df_temperature = pd.DataFrame(data)
        
        intervals = {}
        for i in range(1, 3):  # 2 слоя
            T_intervals = []
            for j in range(len(df_temperature) - 1):
                T_intervals.append((j*25, (j+1)*25))
            intervals[f'T_intervals_{i}'] = T_intervals
        
        # Модель Dutta
        model = DuttaModel()
        
        # Расчет значений
        last_Ro_values, ro_layers = model.calculate_ro_values(intervals, df_temperature)
        
        print("Конечные значения конверсии по слоям:")
        for i, value in enumerate(last_Ro_values):
            print(f"Слой {i+1}: {value:.4f}")
            
    except Exception as e:
        print(f"Ошибка в модели иллита: {e}")
        import traceback
        traceback.print_exc()

def run_biomarkers_example():
    """Пример запуска модели биомаркеров"""
    print("\n=== Модель биомаркеров ===")
    
    try:
        # Тестовые данные
        data = {
            'Время, млн лет': [0, 10, 20, 30]  # 4 точки = 3 интервала
        }
        df_temperature = pd.DataFrame(data)
        
        intervals = {}
        T_intervals = []
        for j in range(len(df_temperature) - 1):
            T_intervals.append((j*15, (j+1)*15))
        intervals['T_intervals_1'] = T_intervals
        
        # Модель биомаркеров
        model = SteraneAromatization_McKenzie1983()
        
        # Расчет
        last_Ro_values, ro_layers = model.calculate_ro_values(intervals, df_temperature)
        
        print(f"Конечное значение конверсии биомаркеров: {last_Ro_values[0]:.4f}")
        
    except Exception as e:
        print(f"Ошибка в модели биомаркеров: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Главная функция для запуска всех моделей"""
    print("Запуск моделей нефтегазогенерации...")
    
    # Запуск примеров
    run_single_component_example()
    run_multi_component_example()
    run_vitrinite_reflectance_example()
    run_illite_conversion_example()
    run_biomarkers_example()
    
    print("\nВсе модели успешно выполнены!")

if __name__ == "__main__":
    main()