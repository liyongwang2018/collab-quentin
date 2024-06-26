"""
The program below is for unit conversion,
including temperature, humidity, pressure

"""


def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert temperature from one unit to another.

    Parameters
    ----------
    value : float
        Numeric value of the temperature to be converted.
    from_unit : str
        The unit of the input temperature (e.g., 'Celsius', 'Fahrenheit', 'Kelvin', 'Rankine').
    to_unit : str
        The desired unit for the converted temperature.
    available_units : tuple
        A tuple containing the available units for conversion, which are 'C', 'F', 'K', and 'R'.

    Returns
    -------
    float
        The converted temperature value.

    Author
    ------
    Liyong Wang
    Date
    2024-02-02
    """
    # Define a dictionary to store conversion factors
    conversion_factors = {
        'Celsius': {'Fahrenheit': lambda x: (x * 9/5) + 32, 'Kelvin': lambda x: x + 273.15, 'Rankine': lambda x: x * 5/9},
        'Fahrenheit': {'Celsius': lambda x: (x - 32) * 5/9, 'Kelvin': lambda x: (x + 459.67) * 5/9, 'Rankine': lambda x: x},
        'Kelvin': {'Celsius': lambda x: x - 273.15, 'Fahrenheit': lambda x: x * 9/5 - 459.67, 'Rankine': lambda x: x * 9/5},
        'Rankine': {'Celsius': lambda x: x * 5/9, 'Fahrenheit': lambda x: x, 'Kelvin': lambda x: x * 5/9},
        'Reaumur': {'Celsius': lambda x: (x - 80) / 2, 'Fahrenheit': lambda x: (x - 459.67) * 5/9, 'Kelvin': lambda x: x - 80},
    }
    # Check if the input units are valid
    

    # Check if the input units are valid
    if from_unit not in conversion_factors or to_unit not in conversion_factors[from_unit]:
        raise ValueError("Invalid units for temperature conversion")

    # Perform the conversion
    converted_temperature = conversion_factors[from_unit][to_unit](value)

    return converted_temperature
    




def convert_pressure(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert pressure from one unit to another.

    Parameters:
    value: Numeric value of the pressure to be converted.
    from_unit: The unit of the input pressure (e.g., 'hPa', 'kPa', 'psi').
    to_unit: The desired unit for the converted pressure.

    available units: 'kPa',  'hPa', 'psi', 'bar', 'atm'

    Returns:
    Converted pressure value.

    Author: Liyong Wang
    Date: 2024-02-02
    """

    # Define a dictionary to store conversion factors
    conversion_factors = {
        'hPa': {
            'kPa': lambda x: x / 10,
            'psi': lambda x: x * 0.0145038,
        },
        'kPa': {
            'hPa': lambda x: x * 10,
            'psi': lambda x: x * 0.1450377377,
        },
        'psi': {
            'hPa': lambda x: x / 0.0145038,
            'kPa': lambda x: x / 0.1450377377,
        },
        'bar': {
            'hPa': lambda x: x * 100,
            'kPa': lambda x: x,
            'psi': lambda x: x * 6894.75729,
        },
        'atm': {
            'hPa': lambda x: x * 101325,
            'kPa': lambda x: x / 101325,
            'psi': lambda x: x * 133.322368,
        },
    }

    # Check if the input units are valid
    if from_unit not in conversion_factors or to_unit not in conversion_factors[from_unit]:
        raise ValueError("Invalid units for pressure conversion")

    # Perform the conversion
    converted_pressure = conversion_factors[from_unit][to_unit](value)

    return converted_pressure


def convert_length(value: float, from_unit: str, to_unit: str) -> float:
    """Convert length from one unit to another.

    Parameters:
    value: Numeric value of the length to be converted.
    from_unit: The unit of the input length (e.g., 'meter', 'kilometer', 'inch').
    to_unit: The desired unit for the converted length.

    available units: 'meter', 'kilometer', 'inch', 'foot','mile

    Returns:
    Converted length value.

    Author: Liyong Wang
    Date: 2024-02-02


    """
    # Define a dictionary to store conversion factors
    conversion_factors = {
        'meter': {
            'kilometer': lambda x: x / 1000,
            'inch': lambda x: x * 39.3700787,
        },
        'kilometer': {
            'meter': lambda x: x * 1000,
            'inch': lambda x: x * 39370.0787,
        },
        'inch': {
            'meter': lambda x: x / 39.3700787,
            'kilometer': lambda x: x / 39370.0787,
        },
        'foot': {
            'meter': lambda x: x / 3.2808399,
            'kilometer': lambda x: x / 3280.8399,
            'inch': lambda x: x * 12,
        },
        'mile': {
            'meter': lambda x: x / 1609.344,
            'kilometer': lambda x: x / 16093.44,
            'inch': lambda x: x * 63360,
        },
    }

    # Check if the input units are valid
    if from_unit not in conversion_factors or to_unit not in conversion_factors[from_unit]:
        raise ValueError("Invalid units for length conversion")

    # Perform the conversion
    converted_length = conversion_factors[from_unit][to_unit](value)

    return converted_length


def convert_weight(value: float, from_unit: str, to_unit: str) -> float:
    """Convert weight from one unit to another.

    Parameters:
    value: Numeric value of the weight to be converted.
    from_unit: The unit of the input weight (e.g., 'gram', 'kilogram', 'pound','ounce','ton').
    to_unit: The desired unit for the converted weight.

    available units: 'gram', 'kilogram', 'pound','ounce','stone','ton'

    Returns:
    Converted weight value.

    author: Liyong Wang
    date: 2024-02-02
    """

    # Define a dictionary to store conversion factors
    conversion_factors = {
        'gram': {
            'kilogram': lambda x: x / 1000,
            'pound': lambda x: x * 0.45359237,
            'ounce': lambda x: x * 28.3495231,
            'stone': lambda x: x * 6.35029318,
            'ton': lambda x: x / 1000000,
        },
        'kilogram': {
            'gram': lambda x: x * 1000,
            'pound': lambda x: x * 2.20462262,
            'ounce': lambda x: x * 35.2739619,
            'stone': lambda x: x * 14.5939029,
            'ton': lambda x: x,
        },
        'pound': {
            'gram': lambda x: x / 0.45359237,
            'kilogram': lambda x: x / 2.20462262,
            'ounce': lambda x: x / 16,
            'stone': lambda x: x / 14,
            'ton': lambda x: x / 2000,
        },
        'ounce': {
            'gram': lambda x: x / 28.3495231,
            'kilogram': lambda x: x / 35.2739619,
            'pound': lambda x: x * 16,
            'stone': lambda x: x / 16,
            'ton': lambda x: x / 200000,
        },
        'stone': {
            'gram': lambda x: x / 6.35029318,
            'kilogram': lambda x: x / 14.5939029,
            'pound': lambda x: x * 14,
            'ounce': lambda x: x * 16,
            'ton': lambda x: x / 1000,
        },
        'ton': {
            'gram': lambda x: x * 1000000,
            'kilogram': lambda x: x,
            'pound': lambda x: x * 2000,
            'ounce': lambda x: x * 28000,
            'stone': lambda x: x * 100,
        },
    }

    # Check if the input units are valid
    if from_unit not in conversion_factors or to_unit not in conversion_factors[from_unit]:
        raise ValueError("Invalid units for weight conversion, valid units are: pound,ounce,kilogram,gram,stone,ton")

    # Perform the conversion
    converted_weight = conversion_factors[from_unit][to_unit](value)

    return converted_weight


def convert_speed(value: float, from_unit: str, to_unit: str) -> float:
    """Convert speed from one unit to another.

    Parameters:
    value: Numeric value of the speed to be converted.
    from_unit: The unit of the input speed (e.g., 'meter per second', 'kilometer per hour', 'mile per hour').
    to_unit: The desired unit for the converted speed.

    available units:'m/s', 'km/h','mph', 'knots'

    Returns:
    Converted speed value.

    
    author: Liyong Wang
    date: 2024-02-02

    """
    conversion_factors = {
        'm/s': {
            'km/h': lambda x: x * 3.6,
            'mph': lambda x: x * 2.23693629,
            'knots': lambda x: x * 1.94384449,
        },
        'km/h': {
            'm/s': lambda x: x / 3.6,
            'mph': lambda x: x / 1.609344,
            'knots': lambda x: x / 1.852,
        },
        'mph': {
            'm/s': lambda x: x / 2.23693629,
            'km/h': lambda x: x * 1.609344,
            'knots': lambda x: x * 1.15077945,
        },
        'knots': {
            'm/s': lambda x: x / 1.94384449,
            'km/h': lambda x: x * 1.852,
            'mph': lambda x: x / 1.15077945,
        },
    }

    if from_unit not in conversion_factors or to_unit not in conversion_factors[from_unit]:
        raise ValueError("Invalid units for speed conversion")

    return conversion_factors[from_unit][to_unit](value)


# Additional Conversion Units
def convert_volume(value: float, from_unit: str, to_unit: str) -> float:
    """Convert volume from one unit to another.

    Parameters:
    value: Numeric value of the volume to be converted.
    from_unit: The unit of the input volume (e.g., 'milliliter', 'liter', 'gallon', 'fluid ounce').
    to_unit: The desired unit for the converted volume.

    Returns:
    Converted volume value.

    """
    conversion_factors = {
        'milliliter': 1,
        'liter': 1000,
        'gallon': 3.785411784,
        'fluid ounce': 29.573529563,
    }

    if from_unit not in conversion_factors or to_unit not in conversion_factors:
        raise ValueError(f"Invalid units for volume conversion: {from_unit} to {to_unit}")

    return value * conversion_factors[to_unit] / conversion_factors[from_unit]


def convert_time(value: float, from_unit: str, to_unit: str) -> float:
    """Convert time from one unit to another.

    Parameters:
    value: Numeric value of the time to be converted.
    from_unit: The unit of the input time (e.g., 'second', 'minute', 'hour', 'day').
    to_unit: The desired unit for the converted time.
    available units:'second', 'minute', 'hour', 'day'

    Returns:
    Converted time value.

    author: Liyong Wang
    date: 2024-02-02



    """
    conversion_factors = {
        'second': 24*60*60,
        'minute': 24*60,
        'hour': 24,
        'day': 1,
    }

    if from_unit not in conversion_factors or to_unit not in conversion_factors:
        raise ValueError(f"Invalid units for time conversion: {from_unit} to {to_unit}")

    return value * conversion_factors[to_unit] / conversion_factors[from_unit]


def convert_energy(value: float, from_unit: str, to_unit: str) -> float:
    """Convert energy from one unit to another.

    Parameters:
    value: Numeric value of the energy to be converted.
    from_unit: The unit of the input energy (e.g., 'joule', 'watt-hour', 'BTU', 'kilowatt-hour', 'megawatt-hour', 'therm').
    to_unit: The desired unit for the converted energy.

    Returns:
    Converted energy value.

    """
    conversion_factors = {
        'joule': 1,
        'watt-hour': 3600,
        'BTU': 1055.056,
        'kilowatt-hour': 1000,
        'megawatt-hour': 1000000,
        'therm': 1000,
    }

    if from_unit not in conversion_factors or to_unit not in conversion_factors:
        raise ValueError(f"Invalid units for energy conversion: {from_unit} to {to_unit}")

    return value * conversion_factors[to_unit] / conversion_factors[from_unit]


def convert_area(value: float, from_unit: str, to_unit: str) -> float:
    """Convert area from one unit to another.

    Parameters:
    value: Numeric value of the area to be converted.
    from_unit: The unit of the input area (e.g., 'square meter', 'square kilometer', 'square inch', 'square foot', 'square mile').
    to_unit: The desired unit for the converted area.
    available units:'square meter','square kilometer','square inch','square foot','square mile'

    Returns:
    Converted area value.

    author: Liyong Wang
    date: 2024-02-02

    """
    # Define a dictionary to store conversion factors
    conversion_factors = {
        'square meter': {
            'square kilometer': lambda x: x / 1000000,
            'square inch': lambda x: x * 1550.003100,
            'square foot': lambda x: x * 10.7639104,
            'square mile': lambda x: x / 2589988.110336,
        },
        'square kilometer': {
            'square meter': lambda x: x * 1000000,
            'square inch': lambda x: x * 1550003100.0031,
            'square foot': lambda x: x * 107639104.0005,
            'square mile': lambda x: x * 2589988.110336,
        },
        'square inch': {
            'square meter': lambda x: x / 1550.003100,
            'square kilometer': lambda x: x / 1550003100.0031,
            'square foot': lambda x: x / 144.0,
            'square mile': lambda x: x / 1968504.4478,
        },
        'square foot': {
            'square meter': lambda x: x / 10.7639104,
            'square kilometer': lambda x: x / 107639104.0005,
            'square inch': lambda x: x * 144.0,
            'square mile': lambda x: x / 27878400.0,
        },
        'square mile': {
            'square meter': lambda x: x * 2589988.110336,
            'square kilometer': lambda x: x * 2589988110.336,
            'square inch': lambda x: x * 1968504447.8,
            'square foot': lambda x: x * 27878400.0,
        },
    }

    # Check if the input units are valid
    if from_unit not in conversion_factors or to_unit not in conversion_factors[from_unit]:
        raise ValueError("Invalid units for area conversion")

    # Perform the conversion
    converted_area = conversion_factors[from_unit][to_unit](value)

    return converted_area


# ... Add more conversion units as needed



