import numpy as np
from typing import List, Union


# ----------------------
# Tokenizers for numbers
# ----------------------

class SMETokenizer():
    """
    Sign-mantissa-exponent tokenizer.

    E.g. 3.14 -> ['+', '314', 'E-2']
    """
    def __init__(
            self, 
            mantissa_range = [0, 9999],
            exponent_range = [-100, 100]
        ):
        self.mantissa_range = mantissa_range
        self.exponent_range = exponent_range
        self.vocabs = ['nan', '+', '-']
        _mantissa = [str(i) for i in range(mantissa_range[0], mantissa_range[1] + 1)]
        _exponent = ['E' + str(i) for i in range(exponent_range[0], exponent_range[1] + 1)]
        self.vocabs.extend(_mantissa)
        self.vocabs.extend(_exponent)

    def encode(self, number: float) -> List[str]:
        number = float(number)
        if np.isnan(number):
            return ['nan', 'nan', 'nan']
        
        if number == 0 or abs(number) < 1e-8:
            return ['+', '0', 'E0']
        
        sign = '+' if number >= 0 else '-'
        number = abs(number)
        max_number = self.mantissa_range[1] * 10 ** self.exponent_range[1]
    
        if number > max_number:
            print('number too large, return the extreme number')
            return [sign, self.mantissa_range[1], 'E' + self.exponent_range[1]]
        
        significant = len(str(self.mantissa_range[1]))
        mantissa,exponent = (f"%.{significant-1}e"%number).split('e')
        i,f = mantissa.split('.') # integer, fraction
        f = f.rstrip('0')
        mantissa = i + f
        exponent = str(int(exponent) - len(f))
        return [sign, mantissa, 'E' + exponent]

    def decode(self, sequence: List[str]) -> float:
        sign,mantissa,exponent = sequence
        if 'nan' in sequence:
            return np.nan
        return (1 if sign == '+' else -1) * int(mantissa) * 10 ** int(exponent.lstrip('E'))


class SymFormerTokenizer():
    """
    SymFormer tokenizer. https://arxiv.org/abs/2205.15764

    E.g. 3.14 -> [0.314, 'E1']
    """
    def __init__(
            self,
            significant = 4,
            exponent_range = [-100, 100]
        ):
        self.significant = significant
        self.exponent_range = exponent_range
        self.vocabs = ['nan']
        _exponent = ['E' + str(i) for i in range(exponent_range[0], exponent_range[1] + 1)]
        self.vocabs.extend(_exponent)
    
    def encode(self, number: float) -> List[Union[float, str]]:
        number = float(number)
        if np.isnan(number):
            return [np.nan, 'nan']
        
        if number == 0 or abs(number) < 1e-8:
            return [0., 'E0']
        
        sign = '' if number >= 0 else '-'
        number = abs(number)
        max_number = 10 ** self.exponent_range[1]

        if number > max_number:
            print('number too large, return the extreme number')
            return [float(sign + '1'), 'E' + self.exponent_range[1]]

        mantissa,exponent = (f"%.{self.significant-1}e"%number).split('e')
        if float(mantissa) == 1:
            exponent = str(int(exponent))
        else:
            i,f = mantissa.split('.') # integer, fraction
            i,f = '0',i+f
            mantissa = '.'.join([i, f])
            exponent = str(int(exponent) + 1)
        return [float(sign + mantissa), 'E' + exponent]

    def decode(self, sequence: List[Union[float, str]]) -> float:
        mantissa,exponent = sequence
        if 'nan' in sequence:
            return np.nan
        return mantissa * 10 ** int(exponent.lstrip('E'))