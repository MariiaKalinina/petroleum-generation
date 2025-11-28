# -*- coding: utf-8 -*-
"""
Petroleum Generation Models Module

This module contains implementations for:
- Single component hydrocarbon generation
- Multi-component hydrocarbon generation  
- Vitrinite reflectance models for thermal maturity
- Illite transformation models
- Biomarker transformation models
"""

import numpy as np
import pandas as pd
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib import cm, colors


def temperature_integral_numerical(T_start, T_end, E_a, R=1.987e-3):
    """
    Calculate the temperature integral for non-isothermal kinetics.
    
    Parameters:
    -----------
    T_start, T_end : float
        Start and end temperatures in Kelvin
    E_a : float or array-like
        Activation energy in kcal/mol
    R : float
        Gas constant (default: 1.987e-3 kcal/(mol·K))
    
    Returns:
    --------
    float
        Temperature integral value
    """
    integrand = lambda T, E_a, R: np.exp(-E_a / (R * T))
    integral_result = quad(integrand, T_start, T_end, args=(E_a, R))[0]
    return integral_result


class SingleComponentGeneration:
    """Single component hydrocarbon generation model"""
    
    def __init__(self):
        self.R = 1.987e-3  # Gas constant, kcal/(mol·K)
    
    def calculate_conversion_non_isothermal(self, S_i0, A, E_a, T_intervals, HRs):
        """
        Calculate total conversion based on provided parameters for non-isothermal kinetics.
        
        Parameters:
        -----------
        S_i0 : array-like
            Initial fractions of reactant components
        A : float
            Pre-exponential factor
        E_a : array-like  
            Activation energies for each component
        T_intervals : list of tuples
            Temperature intervals in Celsius
        HRs : array-like
            Heating rates for each interval
            
        Returns:
        --------
        tuple
            (total_conversion, temperatures, conversions, dataframe)
        """
        conversions = []
        temperatures = []
        current_fracs = np.array(S_i0)
        df = pd.DataFrame(columns=['Myr', 'Temperature Interval', 'Current Conversion'])
        
        Myr_intervals = list(range(len(T_intervals)))  # Placeholder time intervals

        for j in range(len(T_intervals)):
            T_start, T_end = T_intervals[j]
            T_start_k, T_end_k = T_start + 273.15, T_end + 273.15  # Convert to Kelvin
            Hr = HRs[j]

            for i in range(len(S_i0)):
                I_E_T = temperature_integral_numerical(T_start_k, T_end_k, E_a[i], self.R)
                conversion_increment = np.exp(-A / Hr * I_E_T)
                current_fracs[i] *= conversion_increment

            current_conversion = 1 - current_fracs.sum()
            conversions.append(current_conversion)

            new_row = pd.DataFrame({
                'Myr': [Myr_intervals[j]],
                'Temperature Interval': [f"{T_intervals[j][0]} - {T_intervals[j][1]}"],
                'Current Conversion': [f"{current_conversion:.4f}"]
            })
            df = pd.concat([df, new_row], ignore_index=True)

        total_conversion = conversions[-1]
        print(f"Total conversion (α_total): {total_conversion:.4f}")

        return total_conversion, temperatures, conversions, df


class MultiComponentGeneration:
    """Multi-component hydrocarbon generation model"""
    
    def __init__(self):
        self.R = 1.987e-3  # Gas constant, kcal/(mol·K)
    
    def calculate_conversion_non_isothermal_multi(self, S_i0, A, E_a, T_intervals, HRs, Ratio):
        """
        Calculate total conversion for multi-component system.
        
        Parameters:
        -----------
        S_i0 : list of arrays
            Initial fractions for each component
        A : list
            Pre-exponential factors for each component
        E_a : list of arrays
            Activation energies for each component
        T_intervals : list of tuples
            Temperature intervals in Celsius
        HRs : array-like
            Heating rates for each interval
        Ratio : array-like
            Ratios for each component
            
        Returns:
        --------
        list
            Conversion values for each time step
        """
        conversions = []
        current_fracs = np.array(S_i0) * Ratio

        for j in range(len(T_intervals)):
            T_start, T_end = T_intervals[j]
            T_start_k, T_end_k = T_start + 273.15, T_end + 273.15  # Convert to Kelvin
            Hr = HRs[j]

            for i in range(len(S_i0)):
                I_E_T = temperature_integral_numerical(T_start_k, T_end_k, E_a[i], self.R)
                conversion_increment = np.exp(-A / Hr * I_E_T)
                current_fracs[i] *= conversion_increment

            current_conversion = current_fracs.sum()
            conversions.append(current_conversion)

        return conversions
    
    def calculate_conversion_non_isothermal_multi_for_mass_gen(self, S_i0, A, E_a, T_intervals, HRs, Ratio):
        """
        Calculate conversion for mass generation in multi-component system.
        
        Parameters:
        -----------
        S_i0 : array-like
            Initial fractions
        A : float
            Pre-exponential factor
        E_a : array-like
            Activation energies
        T_intervals : list of tuples
            Temperature intervals in Celsius
        HRs : array-like
            Heating rates for each interval
        Ratio : float
            Component ratio
            
        Returns:
        --------
        list
            Conversion values for each time step
        """
        conversions = []
        current_fracs = np.array(S_i0) * Ratio

        for j in range(len(T_intervals)):
            T_start, T_end = T_intervals[j]
            T_start_k, T_end_k = T_start + 273.15, T_end + 273.15  # Convert to Kelvin
            Hr = HRs[j]

            for i in range(len(S_i0)):
                I_E_T = temperature_integral_numerical(T_start_k, T_end_k, E_a[i], self.R)
                conversion_increment = np.exp(-A / Hr * I_E_T)
                current_fracs[i] *= conversion_increment

            current_conversion = Ratio - current_fracs.sum()
            conversions.append(current_conversion)

        return conversions


class VitriniteReflectanceModel:
    """Base class for vitrinite reflectance models"""
    
    def __init__(self, A0, S_i0, E_a, Ro_start, Ro_end):
        self.A0 = A0
        if isinstance(self.A0, (list, np.ndarray)):
            self.A = 3.17 * 10**11 * np.array(A0)
        else:
            self.A = 3.17 * 10**11 * A0
        self.S_i0 = np.array(S_i0)
        self.E_a = np.array(E_a)
        self.Ro_start = Ro_start
        self.Ro_end = Ro_end

    def calculate_Ro(self, F):
        raise NotImplementedError("This method should be implemented by subclasses")

    def temperature_integral_numerical_Ro(self, T_start, T_end, E_a, R=1.987e-3):
        integrand = lambda T, E_a, R: np.exp(-E_a / (R * T))
        integral_result = quad(integrand, T_start, T_end, args=(E_a, R))[0]
        return integral_result

    def calculate_conversion_non_isothermal_Ro(self, T_intervals, HRs, time_steps_interval):
        conversions = []
        current_fracs = self.S_i0.copy()
        
        for j in range(len(T_intervals)):
            T_start, T_end = T_intervals[j]
            T_start_k, T_end_k = T_start + 273.15, T_end + 273.15
            Hr = HRs[j]
            
            for i in range(len(self.S_i0)):
                I_E_T = self.temperature_integral_numerical_Ro(T_start_k, T_end_k, self.E_a[i])
                if isinstance(self.A, np.ndarray):
                    conversion_increment = np.exp(-self.A[i] / Hr * I_E_T)
                else:
                    conversion_increment = np.exp(-self.A / Hr * I_E_T)
                current_fracs[i] *= conversion_increment
                
            current_conversion = 1 - current_fracs.sum()
            conversions.append(current_conversion)
            
        total_conversion = conversions[-1]
        return total_conversion, conversions

    def calculate_ro_values(self, intervals, df_temperature):
        Myr_intervals = df_temperature["Время, млн лет"].tolist()[::-1]
        ro_layers = {}
        last_Ro_values = []

        for idx, (layer_key, T_intervals) in enumerate(intervals.items()):
            # Правильно рассчитываем временные шаги
            time_steps_interval = np.abs(np.diff(Myr_intervals))
            if len(time_steps_interval) < len(T_intervals):
                # Если временных шагов меньше, чем температурных интервалов, дополняем
                time_steps_interval = np.append(time_steps_interval, [time_steps_interval[-1]] * (len(T_intervals) - len(time_steps_interval)))
            
            Hrs_interval = []
            for i in range(len(T_intervals)):
                T_start, T_end = T_intervals[i]
                
                # Обеспечиваем, что индекс не выходит за границы
                if i < len(time_steps_interval):
                    time_step = time_steps_interval[i]
                else:
                    time_step = time_steps_interval[-1] if len(time_steps_interval) > 0 else 1e-20
                    
                if T_end - T_start == 0:
                    T_start += 0.0001
                    Hrs_interval.append((T_end - T_start) / (time_step * 3.15 * 1e13))
                elif time_step == 0:
                    time_step = 1e-20
                    Hrs_interval.append((T_end - T_start) / (time_step * 3.15 * 1e13))
                else:
                    Hrs_interval.append((T_end - T_start) / (time_step * 3.15 * 1e13))

            total_conversion, conversions = self.calculate_conversion_non_isothermal_Ro(
                T_intervals, np.array(Hrs_interval), time_steps_interval
            )
            Ro_layer = self.calculate_Ro(conversions)

            Time_intervals = Myr_intervals[:len(Ro_layer)]
            last_Ro_values.append(Ro_layer[-1])
            ro_layers[layer_key] = (Time_intervals[::-1], Ro_layer)

        return last_Ro_values, ro_layers

    def plot_ro_vs_time(self, intervals, df_temperature):
        last_Ro_values, ro_layers = self.calculate_ro_values(intervals, df_temperature)
        colormap = get_cmap('hsv')
        plt.figure(figsize=(8, 6))
        norm = Normalize(vmin=0, vmax=len(ro_layers) - 1)
        
        for idx, (layer_key, (Time_intervals, Ro_layer)) in enumerate(reversed(list(ro_layers.items()))):
            color = colormap(norm(idx))
            plt.plot(Time_intervals, Ro_layer, color=color, lw=3, marker='o')
            
        sm = cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), ticks=range(len(ro_layers)))
        cbar.set_label("Layer Index")
        plt.xlabel("Time [Myr]")
        plt.ylabel("Conversion [%]")
        plt.title(f"Ro vs Time for Each Layer ({self.__class__.__name__})")
        plt.grid(True)
        plt.show()


class SweeneyBurnhamModel(VitriniteReflectanceModel):
    """Sweeney & Burnham (1990) Easy%Ro model"""
    
    def __init__(self):
        super().__init__(
            A0=31.7,
            S_i0=np.array([3.53, 3.53, 4.71, 4.71, 5.88, 5.88, 7.06, 4.71, 4.71, 8.23, 
                        7.06, 7.06, 7.06, 5.88, 5.88, 4.71, 3.53, 2.35, 2.35, 1.17]) / 100,
            E_a=[34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72],
            Ro_start=0.20,
            Ro_end=4.66
        )

    def calculate_Ro(self, F):
        return self.Ro_start * (self.Ro_end / self.Ro_start) ** np.array(F)


class IlliteConversionModel:
    """Base class for illite conversion models"""
    
    def __init__(self, A0, S_i0, E_a, Ro_start, Ro_end):
        self.A0 = A0
        if isinstance(self.A0, (list, np.ndarray)):
            self.A = 3.17 * 10**11 * np.array(A0)
        else:
            self.A = 3.17 * 10**11 * A0
        self.S_i0 = np.array(S_i0) / 100
        self.E_a = np.array(E_a)
        self.Ro_start = Ro_start
        self.Ro_end = Ro_end

    def temperature_integral_numerical_Ro(self, T_start, T_end, E_a, R=1.987e-3):
        integrand = lambda T, E_a, R: np.exp(-E_a / (R * T))
        integral_result = quad(integrand, T_start, T_end, args=(E_a, R))[0]
        if integral_result == 0:
            integral_result = 1e-30
        return integral_result

    def calculate_conversion_non_isothermal_Ro(self, T_intervals, HRs):
        conversions = []
        current_fracs = self.S_i0.copy()
        
        for j in range(len(T_intervals)):
            T_start, T_end = T_intervals[j]
            T_start_k, T_end_k = T_start + 273.15, T_end + 273.15
            Hr = HRs[j]
            
            for i in range(len(self.S_i0)):
                I_E_T = self.temperature_integral_numerical_Ro(T_start_k, T_end_k, self.E_a[i])
                if isinstance(self.A, np.ndarray):
                    conversion_increment = np.exp(-self.A[i] / Hr * I_E_T)
                else:
                    conversion_increment = np.exp(-self.A / Hr * I_E_T)
                current_fracs[i] *= conversion_increment
                
            current_conversion = 1 - current_fracs.sum()
            conversions.append(current_conversion)
            
        total_conversion = conversions[-1]
        return total_conversion, conversions

    def calculate_Ro(self, F):
        y_min = self.Ro_start
        y_max = self.Ro_end
        normalized_array = y_min + (np.array(F)) * (y_max - y_min)
        return normalized_array


class DuttaModel(IlliteConversionModel):
    """Dutta (1986) illite conversion model"""
    
    def __init__(self):
        super().__init__(
            A0=4 * 1e-15,
            S_i0=np.array([100]),
            E_a=[19.3],
            Ro_start=0.2,
            Ro_end=0.8
        )


class BiomarkersModel:
    """Base class for biomarker transformation models"""
    
    def __init__(self, A0, S_i0, E_a, Ro_start, Ro_end):
        self.A0 = A0
        if isinstance(self.A0, (list, np.ndarray)):
            self.A = 3.17 * 10**11 * np.array(A0)
        else:
            self.A = 3.17 * 10**11 * A0
        self.S_i0 = np.array(S_i0) / 100
        self.E_a = np.array(E_a)
        self.Ro_start = Ro_start
        self.Ro_end = Ro_end

    def calculate_Ro(self, F):
        y_min = self.Ro_start
        y_max = self.Ro_end
        normalized_array = y_min + (np.array(F)) * (y_max - y_min)
        return normalized_array

    def temperature_integral_numerical_Ro(self, T_start, T_end, E_a, R=1.987e-3):
        integrand = lambda T, E_a, R: np.exp(-E_a / (R * T))
        integral_result = quad(integrand, T_start, T_end, args=(E_a, R))[0]
        return integral_result

    def calculate_conversion_non_isothermal_Ro(self, T_intervals, HRs):
        conversions = []
        current_fracs = self.S_i0.copy()
        
        for j in range(len(T_intervals)):
            T_start, T_end = T_intervals[j]
            T_start_k, T_end_k = T_start + 273.15, T_end + 273.15
            Hr = HRs[j]
            
            for i in range(len(self.S_i0)):
                I_E_T = self.temperature_integral_numerical_Ro(T_start_k, T_end_k, self.E_a[i])
                if isinstance(self.A, np.ndarray):
                    conversion_increment = np.exp(-self.A[i] / Hr * I_E_T)
                else:
                    conversion_increment = np.exp(-self.A / Hr * I_E_T)
                current_fracs[i] *= conversion_increment
                
            current_conversion = 1 - current_fracs.sum()
            conversions.append(current_conversion)
            
        total_conversion = conversions[-1]
        return total_conversion, conversions


class SteraneAromatization_McKenzie1983(BiomarkersModel):
    """Sterane aromatization model - McKenzie (1983)"""
    
    def __init__(self):
        super().__init__(
            A0=5.680 * 1e2,
            S_i0=np.array([100]),
            E_a=np.array([47.769]),
            Ro_start=0,
            Ro_end=1
        )


# Export main classes
__all__ = [
    'SingleComponentGeneration',
    'MultiComponentGeneration', 
    'VitriniteReflectanceModel',
    'SweeneyBurnhamModel',
    'IlliteConversionModel',
    'DuttaModel',
    'BiomarkersModel',
    'SteraneAromatization_McKenzie1983',
    'temperature_integral_numerical'
]

