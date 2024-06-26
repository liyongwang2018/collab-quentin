import math


# Ideal Gas Law
def ideal_gas_law(P: float, V: float, n: float, R: float = 8.314) -> float:
    """
    Calculates the temperature T from pressure P, volume V, and amount of moles n using the ideal gas law.

    The ideal gas law states that:

    T = PV / nR

    Where:
        T is the temperature in Kelvin
        P is the pressure in Pascals
        V is the volume in cubic meters
        n is the amount of moles
        R is the universal gas constant (8.314 J/mol K by default)

    Args:
        P (float): Pressure, in Pascals (Pa).
        V (float): Volume, in cubic meters (m^3).
        n (float): Number of moles.
        R (float, optional): Universal gas constant, in Joules per mole Kelvin (J/mol K). Defaults to 8.314.

    Returns:
        float: Temperature, in Kelvin (K).

    Raises:
        ValueError: If the input values are not positive numbers.
    
    Author: Liyong Wang
    Date: 2024-02-03
    """
    if P <= 0 or V <= 0 or n <= 0:
        raise ValueError("All input values must be positive numbers")

    T = P * V / (n * R)
    return T


# Stoichiometry
def stoichiometry(A: float, B: float, a: float, b: float) -> tuple:
    """
    Calculates the moles of C and D from the stoichiometric coefficients.

    Args:
        A (float): Concentration of reactant A.
        B (float): Concentration of reactant B.
        a (float): Stoichiometric coefficient for A.
        b (float): Stoichiometric coefficient for B.

    Returns:
        tuple: A tuple containing the moles of C and D.

    Raises:
        ValueError: If the stoichiometric coefficients are not integers.  

    Equation:
    moles_of_C = (a * A) / a
    moles_of_D = (b * B) / b

    Code:
    if not isinstance(a, int) or not isinstance(b, int):
        raise ValueError("The stoichiometric coefficients must be integers")

    moles_of_C = (a * A) / a
    moles_of_D = (b * B) / b

    return moles_of_C, moles_of_D  
    
    Author: Liyong Wang
    Date: 2024-02-03


    """
    if not isinstance(a, int) or not isinstance(b, int):
        raise ValueError("The stoichiometric coefficients must be integers")

    moles_of_C = (a * A) / a
    moles_of_D = (b * B) / b

    return moles_of_C, moles_of_D


# Conduction Heat Transfer
def conduction_heat_transfer(k, A, delta_T, d):
    """
    Calculates the heat transfer rate for conduction based on the thermal conductivity, 
    area, temperature difference, and distance between the objects.

    The heat transfer rate is calculated as:

    Q = k * A * delta_T / d

    Where:
        Q is the heat transfer rate, in Watts (W)
        k is the thermal conductivity, in Watts per meter Kelvin (W/m K)
        A is the area, in square meters (m^2)
        delta_T is the temperature difference, in Kelvin (K)
        d is the distance between the objects, in meters (m)

    Args:
        k (float): Thermal conductivity, in Watts per meter Kelvin (W/m K).
        A (float): Area, in square meters (m^2).
        delta_T (float): Temperature difference, in Kelvin (K).
        d (float): Distance between the objects, in meters (m).

    Returns:
        float: Heat transfer rate, in Watts (W).

    Raises:
        ValueError: If the input values are not positive numbers.

    Author: Liyong Wang
    Date: 2024-02-03
    """
    if k <= 0 or A <= 0 or delta_T <= 0 or d <= 0:
        raise ValueError("All input values must be positive numbers")

    Q = k * A * delta_T / d
    return Q


# Rate Equation
def rate_equation(k: float, A: float, B: float, m: float, n: float) -> float:
    """
    Calculates the rate constant for a chemical reaction based on the
    Michaelis-Menten equation.

    The rate constant, `r`, is given by the following equation:

    .. math::
        r = k [A^m][B^n]

    Where:

    - :math:`k` is the rate constant
    - :math:`A` and :math:`B` are the concentrations of the reactants
    - :math:`m` and :math:`n` are the stoichiometric coefficients of the reaction

    Args:
        k (float): Rate constant, :math:`k`.
        A (float): Concentration of reactant A, :math:`A`.
        B (float): Concentration of reactant B, :math:`B`.
        m (float): Stoichiometric coefficient for reactant A, :math:`m`.
        n (float): Stoichiometric coefficient for reactant B, :math:`n`.

    Returns:
        float: Rate constant, :math:`r`.

    Raises:
        ValueError: If the stoichiometric coefficients are not integers.
    
    Author: Liyong Wang
    Date: 2024-02-03

    """
    if not isinstance(m, int) or not isinstance(n, int):
        raise ValueError("The stoichiometric coefficients must be integers")

    r = k * (A ** m) * (B ** n)
    return r


# Mass Balance
def mass_balance(input_mass: float, output_mass: float, accumulation_rate: float) -> bool:
    """
    Calculates whether the input mass is equal to the output mass plus the accumulation rate.

    The mass balance equation is:

    input_mass = output_mass + accumulation_rate

    Args:
        input_mass (float): The mass entering the system.
        output_mass (float): The mass exiting the system.
        accumulation_rate (float): The mass accumulating in the system.

    Returns:
        bool: Whether the input mass is equal to the output mass plus the accumulation rate.

    Raises:
        ValueError: If the input mass, output mass, or accumulation rate are negative numbers.
    
    Author: Liyong Wang
    Date: 2024-02-03

    """
    if input_mass < 0 or output_mass < 0 or accumulation_rate < 0:
        raise ValueError("All input masses must be non-negative numbers")

    return input_mass == output_mass + accumulation_rate


# Raoult's Law
def raoults_law(x_i, P_i_star):
    """
    Calculates the vapor pressure of a liquid based on the Raoult's Law.

    The vapor pressure, `P_i`, is given by the following equation:

    .. math::
        P_i = x_i P_i^*

    Where:

    - :math:`x_i` is the mole fraction of the component in the vapor phase
    - :math:`P_i^*` is the vapor pressure of the pure component

    Args:
        x_i (float): Mole fraction of the component in the vapor phase.
        P_i_star (float): Vapor pressure of the pure component.

    Returns:
        float: Vapor pressure of the mixture, `P_i`.

    Raises:
        ValueError: If the input values are not positive numbers.

    Author: Liyong Wang
    Date: 2024-02-03
    """
    if x_i <= 0 or P_i_star <= 0:
        raise ValueError("All input values must be positive numbers")

    P_i = x_i * P_i_star
    return P_i


# Nernst Equation
def nernst_equation(E_standard, n, concentrations):
    """
    Calculates the Nernst Equation for a chemical reaction.

    The Nernst Equation is used to calculate the equilibrium potential between two
    chemical species in a reaction, given the standard reduction potential (E_standard),
    the stoichiometric coefficients (a, b, c, and d), and the concentrations of the reactants
    (A, B, C, and D).

    The Nernst Equation is given by:

    E = E_standard - (0.0592 / n) * ln((C^c * D^d) / (A^a * B^b))

    Where:

    E: The equilibrium potential, in volts (V).
    E_standard: The standard reduction potential, in volts (V).
    n: The number of electrons transferred in the reaction.
    C, D, A, and B: The concentrations of the reactants, in moles per liter (mol/L).
    c and d: The stoichiometric coefficients of the reaction.

    Returns:
    The equilibrium potential, E, as a float.

    Raises:
    ValueError: If the input values are not positive numbers.
    
    Author: Liyong Wang
    Date: 2024-02-03

    """
    if n <= 0 or E_standard <= 0:
        raise ValueError("All input values must be positive numbers")

    c, d = stoichiometry(A=concentrations['C'], B=concentrations['D'], a=c, b=d)
    a, b = stoichiometry(A=concentrations['A'], B=concentrations['B'], a=a, b=b)

    numerator = c ** c * d ** d
    denominator = a ** a * b ** b
    E = E_standard - (0.0592 / n) * math.log(numerator / denominator)
    return E


# Reynolds Number
def reynolds_number(rho: float, u: float, L: float, mu: float) -> float:
    """
    Calculates the Reynolds number based on the velocity, length, and viscosity of the fluid.

    The Reynolds number is defined as:

    Re = ρ * u * L / μ

    Where:

    Re: The Reynolds number.
    rho: The density of the fluid, in kilograms per cubic meter (kg/m^3).
    u: The velocity of the fluid, in meters per second (m/s).
    L: The characteristic length of the system, in meters (m).
    mu: The viscosity of the fluid, in kilograms per meter second (kg/m s).

    Returns:
    The Reynolds number, Re, as a float.

    Raises:
    ValueError: If any of the input values are negative numbers.
    
    Author: Liyong Wang
    Date: 2024-02-03

    """
    if rho <= 0 or u <= 0 or L <= 0 or mu <= 0:
        raise ValueError("All input values must be positive numbers")

    Re = rho * u * L / mu
    return Re


# CSTR Residence Time
def cstr_residence_time(V: float, FA0: float) -> float:
    """
    Calculates the residence time of a continuous stirred tank reactor (CSTR) based on the volume and the feed flow rate.

    The residence time, :math:`\tau`, is given by the following equation:

    .. math::
        \tau = \frac{V}{F_{A0}}

    Where:

    - :math:`\tau` is the residence time
    - `V` is the volume of the reactor, in cubic meters (m^3)
    - `F_{A0}` is the feed flow rate, in cubic meters per hour (m^3/h)

    Returns:
    The residence time, :math:`\tau`, as a float.

    Raises:
    ValueError: If either the volume or the feed flow rate are non-positive numbers.

    Author: Liyong Wang
    Date: 2024-02-03

    """
    if V <= 0 or FA0 <= 0:
        raise ValueError("Volume and feed flow rate must be positive numbers")

    tau = V / FA0
    return tau


# Van der Waals Equation
def van_der_waals_equation(P: float, V: float, n: float, T: float, a: float, b: float) -> float:
    """
    Calculates the Van der Waals equation for a system of interacting particles.

    The Van der Waals equation describes the intermolecular forces between particles in a system,
    and is given by:

    .. math::
        U(P, V, n, T, a, b) = P V - n b + \frac{n \times 8.314 \times T}{1}

    Where:

    - :math:`U(P, V, n, T, a, b)` is the Van der Waals energy
    - :math:`P` is the pressure
    - :math:`V` is the volume
    - :math:`n` is the total number of particles
    - :math:`T` is the temperature
    - :math:`a` is the parameter that determines the strength of the attractive force
    - :math:`b` is the parameter that determines the strength of the repulsive force

    The equation can be used to calculate the equilibrium pressure and volume of a system of
    particles, as well as the temperature at which the system reaches equilibrium.

    Args:
        P (float): Pressure, in Pascals (Pa).
        V (float): Volume, in cubic meters (m^3).
        n (float): Number of particles.
        T (float): Temperature, in Kelvin (K).
        a (float): Attractive force parameter.
        b (float): Repulsive force parameter.

    Returns:
        float: The Van der Waals energy, in Joules (J).

    Raises:
        ValueError: If the input values are not positive numbers.
    
    Author: Liyong Wang
    Date: 2024-02-03

    """
    if P <= 0 or V <= 0 or n <= 0:
        raise ValueError("All input values must be positive numbers")

    corrected_P = P + (a * n ** 2) / V ** 2
    corrected_V = V - n * b
    equation_result = corrected_P * corrected_V - n * 8.314 * T
    return equation_result


# Arrhenius Equation
def arrhenius_equation(A: float, E_a: float, T: float) -> float:
    """
    Calculates the activation energy for an Arrhenius reaction based on the
    Arrhenius equation.

    The activation energy, `E_a`, is given by the following equation:

    .. math::
        E_a = A / \ln(\frac{k_0}{k})

    Where:

    - :math:`A` is the pre-exponential factor
    - :math:`k_0` is the original rate constant at infinite temperature
    - :math:`k` is the rate constant at the given temperature, :math:`T`

    Args:
        A (float): Pre-exponential factor, :math:`A`.
        E_a (float): Activation energy, :math:`E_a`.
        T (float): Temperature, :math:`T`.

    Returns:
        float: Activation energy, :math:`E_a`.

    Raises:
        ValueError: If the input values are not positive numbers.

    Author: Liyong Wang
    Date: 2024-02-03
    """
    if A <= 0 or E_a <= 0:
        raise ValueError("All input values must be positive numbers")

    k = A * math.exp(-E_a / (8.314 * T))
    return k


# Batch Reactor Conversion
def batch_reactor_conversion(k: float, t: float) -> float:
    """
    Calculates the conversion of a batch reactor based on the rate constant and time.

    The conversion, `X`, is given by the following equation:

    .. math::
        X = 1 - e^{-\frac{k}{t}}

    Where:

    - :math:`X` is the conversion
    - :math:`k` is the rate constant
    - :math:`t` is the time

    Args:
        k (float): Rate constant, :math:`k`.
        t (float): Time, :math:`t`.

    Returns:
        float: Conversion, :math:`X`.

    Raises:
        ValueError: If the input values are not positive numbers.

    Author: Liyong Wang
    Date: 2024-02-03
    """
    if k <= 0 or t <= 0:
        raise ValueError("All input values must be positive numbers")

    X = 1 - math.exp(-k * t)
    return X


# Fugacity Calculation using Virial Equation
def fugacity_virial_equation(B: float, P: float, T: float) -> float:
    """
    Calculates the fugacity coefficient using the Virial Equation.

    The fugacity coefficient, `f`, is given by the following equation:

    .. math::
        f = e^{B P / (R T)}

    Where:

    - :math:`f` is the fugacity coefficient
    - :math:`B` is the virial coefficient
    - :math:`P` is the pressure
    - :math:`R` is the ideal gas constant
    - :math:`T` is the temperature

    Args:
        B (float): Virial coefficient, :math:`B`.
        P (float): Pressure, :math:`P`.
        T (float): Temperature, :math:`T`.

    Returns:
        float: Fugacity coefficient, :math:`f`.

    Raises:
        ValueError: If the input values are not positive numbers.
    
    Author: Liyong Wang
    Date: 2024-02-03

    """
    if B <= 0 or P <= 0 or T <= 0:
        raise ValueError("All input values must be positive numbers")

    f = math.exp(B * P / (8.314 * T))
    return f


# Mole Fraction Calculation
def mole_fraction(n_i, total_moles):
    """
    Calculates the mole fraction of a component in a mixture based on the 
    mole amount and the total mole amount of the mixture.

    The mole fraction, `y_i`, is given by the following equation:

    .. math::
        y_i = n_i / M

    Where:

    - :math:`y_i` is the mole fraction of component i
    - :math:`n_i` is the mole amount of component i
    - :math:`M` is the total mole amount of the mixture

    Args:
        n_i (float): Mole amount of component i, :math:`n_i`.
        total_moles (float): Total mole amount of the mixture, :math:`M`.

    Returns:
        float: Mole fraction of component i, :math:`y_i`.

    Raises:
        ValueError: If the input values are not positive numbers.

    Author: Liyong Wang
    Date: 2024-02-03

    """
    if n_i <= 0 or total_moles <= 0:
        raise ValueError("All input values must be positive numbers")

    y_i = n_i / total_moles
    return y_i


# McCabe-Thiele Method
def mccabe_thiele_method(L: float, V: float, x_D: float, y_D: float) -> tuple:
    """
    Calculates the equilibrium composition and concentration of a 
    packed bed reactor using the McCabe-Thiele method.

    The McCabe-Thiele method is a method for calculating the equilibrium 
    composition and concentration of a packed bed reactor. It assumes 
    that the reactor is in a steady state and that the mass transfer 
    rate is proportional to the square of the distance between the 
    reactants.

    The equilibrium composition and concentration are calculated based 
    on the following equations:

    - Equilibrium composition:

    .. math::
        C_i = \frac{y_i V}{L}

    - Equilibrium concentration:

    .. math::
        \frac{C_i}{C_{total}} = \frac{y_i}{\sum_{j=1}^n y_j}

    Where:

    - :math:`C_i` is the concentration of component i
    - :math:`y_i` is the mole fraction of component i
    - :math:`V` is the volume of the packed bed reactor
    - :math:`L` is the length of the packed bed reactor
    - :math:`C_{total}` is the total concentration of all components

    Args:
        L (float): Length of the packed bed reactor, :math:`L`.
        V (float): Volume of the packed bed reactor, :math:`V`.
        x_D (float): Initial mole fraction of the dissolved component, :math:`x_D`.
        y_D (float): Initial mole fraction of the dissolved component, :math:`y_D`.

    Returns:
        tuple: A tuple containing the equilibrium composition and concentration of the packed bed reactor.

    Raises:
        ValueError: If the input values are not positive numbers.
    
    Author: Liyong Wang
    Date: 2024-02-03

    """
    if L <= 0 or V <= 0 or x_D <= 0 or y_D <= 0:
        raise ValueError("All input values must be positive numbers")

    C_i = (y_D * V) / L
    C_total = C_i / (x_D + y_D)

    return C_i, C_total


# Packed Bed Reactor Performance
def packed_bed_reactor_conversion(k: float, a: float, epsilon: float, rho: float, Cp: float, U: float) -> float:
    """
    Calculates the conversion of a packed bed reactor based on the rate constant, 
    packing factor, porosity, specific heat capacity, and fluid velocity.

    The conversion, `X`, is given by the following equation:

    .. math::
        X = 1 - \frac{1}{1 + \frac{k a}{\epsilon \rho C_p U}}

    Where:

    - :math:`X` is the conversion
    - :math:`k` is the rate constant
    - :math:`a` is the packing factor
    - :math:`\epsilon` is the porosity
    - :math:`\rho` is the density of the fluid
    - :math:`C_p` is the specific heat capacity of the fluid
    - :math:`U` is the fluid velocity

    Args:
        k (float): Rate constant, :math:`k`.
        a (float): Packing factor, :math:`a`.
        epsilon (float): Porosity, :math:`\epsilon`.
        rho (float): Density of the fluid, :math:`\rho`.
        Cp (float): Specific heat capacity of the fluid, :math:`C_p`.
        U (float): Fluid velocity, :math:`U`.

    Returns:
        float: Conversion, :math:`X`.

    Raises:
        ValueError: If any of the input values are negative numbers.

    """
    if k <= 0 or a <= 0 or epsilon <= 0 or rho <= 0 or Cp <= 0 or U <= 0:
        raise ValueError("All input values must be positive numbers")

    X = 1 - 1 / (1 + (k * a) / (epsilon * rho * Cp * U))
    return X


def convection_heat_transfer(h: float, A: float, delta_T: float) -> float:
    """
    Calculates the heat transfer rate for convection based on the heat transfer coefficient, 
    area, temperature difference, and distance between the objects.

    The heat transfer rate is calculated as:

    Q = h * A * delta_T

    Where:
        Q is the heat transfer rate, in Watts (W)
        h is the heat transfer coefficient, in Watts per meter squared (W/m^2)
        A is the area, in square meters (m^2)
        delta_T is the temperature difference, in Kelvin (K)

    Args:
        h (float): Heat transfer coefficient, in Watts per meter squared (W/m^2).
        A (float): Area, in square meters (m^2).
        delta_T (float): Temperature difference, in Kelvin (K).

    Returns:
        float: Heat transfer rate, in Watts (W).

    Raises:
        ValueError: If the input values are not positive numbers.
    
    Author: Liyong Wang
    Date: 2024-02-03
    """
    if h <= 0 or A <= 0 or delta_T <= 0:
        raise ValueError("All input values must be positive numbers")

    Q = h * A * delta_T
    return Q


def effectiveness_NTU_method(NTU: float, C_r: float) -> float:
    """
    Calculates the effectiveness of a non-woven fabric based on the
    number of threads per unit area (NTU) and the capture rate (C_r).

    The effectiveness, `e`, is given by the following equation:

    .. math::
        e = 1 - e^{-\frac{NTU}{C_r}}

    Where:

    - :math:`e` is the effectiveness
    - :math:`NTU` is the number of threads per unit area
    - :math:`C_r` is the capture rate

    Args:
        NTU (float): Number of threads per unit area, :math:`NTU`.
        C_r (float): Capture rate, :math:`C_r`.

    Returns:
        float: Effectiveness, :math:`e`.

    Raises:
        ValueError: If either the NTU or the C_r are non-positive numbers.
    
    Author: Liyong Wang
    Date: 2024-02-03

    """
    if NTU <= 0 or C_r <= 0:
        raise ValueError("NTU and C_r must be positive numbers")

    effectiveness = 1 - math.exp(-NTU / C_r)
    return effectiveness


def stefan_boltzmann_law(sigma: float, A: float, T1: float, T2: float) -> float:
    """
    Calculates the Stefan-Boltzmann law for radiative heat transfer.

    The Stefan-Boltzmann law states that the total radiative heat transfer, Q, is directly proportional to the fourth power of the absolute temperature difference, T1 - T2, and the emissivity, sigma, of the surface:

    Q = sigma * A * (T1**4 - T2**4)

    Where:
        Q is the total radiative heat transfer, in Watts (W)
        sigma is the emissivity of the surface, between 0 and 1
        A is the surface area, in square meters (m^2)
        T1 and T2 are the absolute temperatures of the emitting and absorbing objects, in Kelvin (K)

    Returns:
        float: The total radiative heat transfer, in Watts (W).

    Raises:
        ValueError: If the input values are not positive numbers.
    
    Author: Liyong Wang
    Date: 2024-02-03

    """
    if sigma <= 0 or A <= 0 or T1 <= 0 or T2 <= 0:
        raise ValueError("All input values must be positive numbers")

    Q = sigma * A * (T1 ** 4 - T2 ** 4)
    return Q


def log_mean_temperature_difference(delta_T1: float, delta_T2: float) -> float:
    """
    Calculates the Log Mean Temperature Difference (LMTD) between two temperatures.

    The LMTD is defined as the temperature difference between the two temperatures,
    divided by the natural logarithm of the ratio of those two temperatures.

    The LMTD is a useful metric for comparing the temperature differences between
    two systems, and can be used to evaluate the stability of a system or the
    effectiveness of a heat transfer process.

    The LMTD is calculated as:

    LMTD = (delta_T1 - delta_T2) / log(delta_T1 / delta_T2)

    Where:

    LMTD is the Log Mean Temperature Difference, in Kelvin (K)
    delta_T1 and delta_T2 are the two temperatures being compared, in Kelvin (K)

    Returns:

    The LMTD between the two temperatures, as a float.

    Raises:

    ValueError: If either delta_T1 or delta_T2 are non-positive numbers.
    
    Author: Liyong Wang
    Date: 2024-02-03

    """
    if delta_T1 <= 0 or delta_T2 <= 0:
        raise ValueError("Both temperatures must be positive numbers")

    LMTD = (delta_T1 - delta_T2) / math.log(delta_T1 / delta_T2)
    return LMTD


def extended_surface_heat_transfer(
        h: float,  # heat transfer coefficient, in W/m^2
        P: float,  # pressure, in Pa
        k: float,  # thermal conductivity, in W/m K
        A_c: float,  # cross-sectional area of the cold surface, in m^2
        m: float,  # slope of the cold surface, in degrees
        C: float,  # specific heat capacity of the fluid, in J/kg K
        A_f: float,  # cross-sectional area of the fluid, in m^2
        L: float,  # characteristic length of the system, in m
        delta_T: float,  # temperature difference, in K
) -> float:  # heat transfer rate, in W
    """
    Calculates the heat transfer rate for extended surface heat transfer based on the
    Fick's law of diffusion, the Sherwood number, and the Higbie's penetration theory.

    The heat transfer rate is calculated as:

    .. math::
        Q = \\sqrt{(h P k A_c) / (m C A_f)} * \\tanh(m L) * delta_T

    Where:

    - h: heat transfer coefficient, in W/m^2
    - P: pressure, in Pa
    - k: thermal conductivity, in W/m K
    - A_c: cross-sectional area of the cold surface, in m^2
    - m: slope of the cold surface, in degrees
    - C: specific heat capacity of the fluid, in J/kg K
    - A_f: cross-sectional area of the fluid, in m^2
    - L: characteristic length of the system, in m
    - delta_T: temperature difference, in K

    Returns:
    The heat transfer rate, in W.

    Raises:
    ValueError: If any of the input values are negative numbers.
    
    Author: Liyong Wang
    Date: 2024-02-03
    """
    if h <= 0 or P <= 0 or k <= 0 or A_c <= 0 or m <= 0 or C <= 0 or A_f <= 0 or L <= 0 or delta_T <= 0:
        raise ValueError("All input values must be positive numbers")

    Q = math.sqrt((h * P * k * A_c) / (m * C * A_f)) * math.tanh(m * L) * delta_T
    return Q


def fick_diffusion_law(D: float, delta_C: float, delta_x: float) -> float:
    """
    Calculates the Fick's diffusion law for mass transfer.

    The Fick's diffusion law states that the net flux of mass across a
    diffusion boundary is equal to the product of the diffusion coefficient,
    D, and the difference in concentration, delta_C, divided by the distance,
    delta_x:

    J = -D * (delta_C / delta_x)

    Where:

    J: net flux of mass, in kilograms per meter squared per second (kg/m^2 s)
    D: diffusion coefficient, in meters squared per second (m^2 s^-1)
    delta_C: difference in concentration, in kilograms per cubic meter (kg/m^3)
    delta_x: distance, in meters (m)

    Returns:
    The net flux of mass, J, as a float.

    Raises:
    ValueError: If any of the input values are negative numbers.
    
    Author: Liyong Wang
    Date: 2024-02-03

    """
    if D <= 0 or delta_C <= 0 or delta_x <= 0:
        raise ValueError("All input values must be positive numbers")

    J = -D * (delta_C / delta_x)
    return J


def sherwood_number(K: float, L: float, D: float) -> float:
    """
    Calculates the Sherwood number for a mass transfer problem.

    The Sherwood number, `Sh`, is defined as:

    .. math::
        Sh = K L / D

    Where:

    - :math:`Sh` is the Sherwood number
    - :math:`K` is the mass transfer coefficient
    - :math:`L` is the characteristic length of the system
    - :math:`D` is the diffusion coefficient

    Args:
        K (float): Mass transfer coefficient, :math:`K`.
        L (float): Characteristic length of the system, :math:`L`.
        D (float): Diffusion coefficient, :math:`D`.

    Returns:
        float: Sherwood number, :math:`Sh`.

    Raises:
        ValueError: If any of the input values are negative numbers.
    
    Author: Liyong Wang
    Date: 2024-02-03

    """
    if K <= 0 or L <= 0 or D <= 0:
        raise ValueError("All input values must be positive numbers")

    Sh = K * L / D
    return Sh


def mass_transfer_coefficient_from_Sherwood(Sh: float, D: float, L: float) -> float:
    """
    Calculates the mass transfer coefficient from the Sherwood number.

    The mass transfer coefficient, `k`, is given by the following equation:

    .. math::
        k = Sh D / L

    Where:

    - :math:`k` is the mass transfer coefficient
    - :math:`Sh` is the Sherwood number
    - :math:`D` is the diffusion coefficient
    - :math:`L` is the characteristic length of the system

    Args:
        Sh (float): Sherwood number, :math:`Sh`.
        D (float): Diffusion coefficient, :math:`D`.
        L (float): Characteristic length of the system, :math:`L`.

    Returns:
        float: Mass transfer coefficient, :math:`k`.

    Raises:
        ValueError: If any of the input values are negative numbers.
    
    Author: Liyong Wang
    Date: 2024-02-03
    """
    if Sh <= 0 or D <= 0 or L <= 0:
        raise ValueError("All input values must be positive numbers")

    k = Sh * D / L
    return k


def higbies_penetration_theory(D: float, t: float, C_s: float) -> float:
    """
    Calculates the Higbie's penetration theory for mass transfer through a porous medium.

    The Higbie's penetration theory states that the effective diffusion coefficient, `N`,
    is given by the following equation:

    .. math::
        N = \\frac{2 \\sqrt{D / t} \\sqrt{C_s}}{\sqrt{\\pi}}

    Where:

    - `N`: effective diffusion coefficient
    - `D`: diffusion coefficient
    - `t`: tortuosity
    - `C_s`: specific capacity of the porous medium

    Returns:
    The effective diffusion coefficient, `N`, as a float.

    Raises:
    ValueError: If any of the input values are negative numbers.

    Author: Liyong Wang
    Date: 2024-02-03

    """
    if D <= 0 or t <= 0 or C_s <= 0:
        raise ValueError("All input values must be positive numbers")

    N = (2 / math.sqrt(math.pi)) * math.sqrt(D / t) * C_s
    return N


def mass_transfer_porous_medium(
        D: float,  # diffusion coefficient, in m^2/s
        epsilon: float,  # porosity, between 0 and 1
        delta_C: float,  # difference in concentration, in kg/m^3
        delta_x: float,  # distance, in m
) -> float:  # effective diffusion coefficient, in m^2/s
    """
    Calculates the effective diffusion coefficient for mass transfer through a porous medium.

    The effective diffusion coefficient, `N`, is given by the following equation:

    .. math::
        N = \\frac{D \\times \\epsilon}{\\delta_C \\times \\delta_x}

    Where:

    - `N`: effective diffusion coefficient
    - `D`: diffusion coefficient
    - `\epsilon`: porosity
    - `\delta_C`: difference in concentration
    - `\delta_x`: distance

    Returns:
    The effective diffusion coefficient, `N`, as a float.

    Raises:
    ValueError: If any of the input values are negative numbers.
    
    Author: Liyong Wang
    Date: 2024-02-03

    """
    if D <= 0 or epsilon <= 0 or delta_C <= 0 or delta_x <= 0:
        raise ValueError("All input values must be positive numbers")

    N = (D * epsilon) / (delta_C * delta_x)
    return N


def overall_mass_transfer_coefficient_absorption(a: float, k: float, H_a: float, H_g: float) -> float:
    """
    Calculates the overall mass transfer coefficient for an absorption column based on the
    Hagen-Poiseuille equation and the Darcy-Weisbach equation.

    The overall mass transfer coefficient, `K_ya`, is given by the following equation:

    .. math::
        K_{ya} = \\frac{a k}{H_{a} + H_{g}}

    Where:

    - `K_ya`: overall mass transfer coefficient
    - `a`: mass transfer area of the absorption column
    - `k`: mass transfer coefficient of the absorption column
    - `H_a`: Hagen-Poiseuille coefficient for the absorption column
    - `H_g`: Darcy-Weisbach coefficient for the absorption column

    Returns:
    The overall mass transfer coefficient, `K_ya`, as a float.

    Raises:
    ValueError: If any of the input values are negative numbers.
    
    Author: Liyong Wang
    Date: 2024-02-03

    """
    if a <= 0 or k <= 0 or H_a <= 0 or H_g <= 0:
        raise ValueError("All input values must be positive numbers")

    Kya = a * k / (1 / H_a + 1 / H_g)
    return Kya


def hagen_poiseuille_equation(L, mu, Q, r):
    delta_P = (4 * L * mu * Q) / (math.pi * r ** 4)
    return delta_P


def darcy_weisbach_equation(f, L, rho, V, D):
    delta_P = (f * L * rho * V ** 2) / (2 * D)
    return delta_P


def reynolds_number_pipe(rho, V, D, mu):
    Re = (rho * V * D) / mu
    return Re


def ergun_equation(mu: float, epsilon: float, L: float, V: float, d_p: float, rho: float) -> float:
    """
    Calculates the Ergun equation for the pressure drop across a packed bed of granular
    solids.

    The Ergun equation is used to calculate the pressure drop across a packed bed of granular
    solids, and is given by the following equation:

    .. math::
        \Delta P = \frac{150 \mu (1 - \epsilon)^2 L V}{(\epsilon^3 d_p^2) + (1.75 (1 - \epsilon) \rho V^2) / \epsilon d_p} + (1.75 \rho V^2) / \epsilon d_p

    Where:

    - :math:`\Delta P` is the pressure drop, in Pascals (Pa)
    - :math:`\mu` is the dynamic viscosity of the fluid, in Pascals (Pa) seconds (s)
    - :math:`\epsilon` is the porosity of the packed bed
    - :math:`L` is the characteristic length of the packed bed, in meters (m)
    - :math:`V` is the fluid velocity, in meters per second (m/s)
    - :math:`d_p` is the particle diameter, in meters (m)
    - :math:`\rho` is the density of the fluid, in kilograms per cubic meter (kg/m^3)

    Returns:
    The pressure drop, :math:`\Delta P`, as a float.

    Raises:
    ValueError: If any of the input values are negative numbers.

    Author: Liyong Wang
    Date: 2024-02-03

    """
    if mu <= 0 or epsilon <= 0 or L <= 0 or V <= 0 or d_p <= 0 or rho <= 0:
        raise ValueError("All input values must be positive numbers")

    delta_P = (150 * mu * (1 - epsilon) ** 2 * L * V) / (
                (epsilon ** 3 * d_p ** 2) + (1.75 * (1 - epsilon) * rho * V ** 2) / (epsilon * d_p)) + (
                          1.75 * rho * V ** 2) / (epsilon * d_p)
    return delta_P


def toricellis_law(g: float, h: float) -> float:
    """
    Calculates the volume of a torus based on the parameters g (circumradius) and h (height).

    The volume, V, is given by the following equation:

    .. math::
        V = \\sqrt{(2gh)}

    Where:

    - V is the volume of the torus
    - g is the circumradius
    - h is the height

    Args:
        g (float): Circumradius, :math:`g`.
        h (float): Height, :math:`h`.

    Returns:
        float: Volume, :math:`V`.

    Raises:
        ValueError: If either g or h are non-positive numbers.
    
    Author: Liyong Wang
    Date: 2024-02-03

    """
    if g <= 0 or h <= 0:
        raise ValueError("Both g and h must be positive numbers")

    V = math.sqrt(2 * g * h)
    return V
