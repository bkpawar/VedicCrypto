import re
import math
import numpy as np
from sympy import symbols, expand, Poly
import sympy as sp
import matplotlib.pyplot as plt
from pyfiglet import Figlet
import pyfiglet
from colorama import Fore

f = Figlet(font='slant', width=300)
print(Fore.BLUE + f.renderText('equation generator'))


fg = 0
n = input(f"{Fore.LIGHTYELLOW_EX}mantra")
v = 20


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# ############################################translotor#############################################################################
def reverse(lst):
    nps = 0
    new_lst = nps[::-1]
    if isinstance(lst, int):
        new_lst = ''.join(new_lst)
    elif isinstance(lst, str):
        new_lst = ''.join(new_lst)
    elif isinstance(lst, float):
        new_lst = ''.join(new_lst)
    return new_lst


def latin(text):
    conversiontable = {'ॐ': 'oṁ', 'ऀ': 'ṁ', 'ँ': 'ṃ', 'ं': 'ṃ', 'ः': 'ḥ', 'अ': 'a', 'आ': 'ā', 'इ': 'i', 'ई': 'ī',
                       'उ': 'u', 'ऊ': 'ū', 'ऋ': 'r̥', 'ॠ': ' r̥̄', 'ऌ': 'l̥', 'ॡ': ' l̥̄', 'ऍ': 'ê', 'ऎ': 'e', 'ए': 'e',
                       'ऐ': 'ai', 'ऑ': 'ô', 'ऒ': 'o', 'ओ': 'o', 'औ': 'au', 'ा': 'ā', 'ि': 'i', 'ी': 'ī', 'ु': 'u',
                       'ू': 'ū', 'ृ': 'r̥', 'ॄ': ' r̥̄', 'ॢ': 'l̥', 'ॣ': ' l̥̄', 'ॅ': 'ê', 'े': 'e', 'ै': 'ai',
                       'ॉ': 'ô', 'ो': 'o', 'ौ': 'au', 'क़': 'q', 'क': 'k', 'ख़': 'x', 'ख': 'kh', 'ग़': 'ġ', 'ग': 'g',
                       'ॻ': 'g', 'घ': 'gh', 'ङ': 'ṅ', 'च': 'c', 'छ': 'ch', 'ज़': 'z', 'ज': 'j', 'ॼ': 'j', 'झ': 'jh',
                       'ञ': 'ñ', 'ट': 'ṭ', 'ठ': 'ṭh', 'ड़': 'ṛ', 'ड': 'ḍ', 'ॸ': 'ḍ', 'ॾ': 'd', 'ढ़': 'ṛh', 'ढ': 'ḍh',
                       'ण': 'ṇ', 'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n', 'प': 'p', 'फ़': 'f', 'फ': 'ph',
                       'ब': 'b', 'ॿ': 'b', 'भ': 'bh', 'म': 'm', 'य': 'y', 'र': 'r', 'ल': 'l', 'ळ': 'ḷ', 'व': 'v',
                       'श': 'ś', 'ष': 'ṣ', 'स': 's', 'ह': 'h', 'ऽ': '\'', '्': '', '़': '', '०': '0', '१': '1',
                       '२': '2', '३': '3', '४': '4', '५': '5', '६': '6', '७': '7', '८': '8', '९': '9', 'ꣳ': 'ṁ',
                       '।': '.', '॥': '..', ' ': ' ', }
    latin_text = ""

    for char in text:
        latin_text += conversiontable.get(char, char) + ', '

    return latin_text

# ##############################################remove UNNESSARRY#######################################################


def vowels(devanagari_text):
    pattern = r"[ा-ौ]"
    devanagari_text = re.sub(pattern, '', devanagari_text)
    return devanagari_text


def remove(text):
    s = re.sub('क्ष', 'ष', text)
    s = re.sub('श्र', 'र', s)
    s = re.sub('त्र', 'त', s)
    s = re.sub('ज्ञ', 'ञ', s)
    nx = separate_half_letters(s)
    a = nx
    a = vowels(a)
    return a


def separate_half_letters(devanagari_word):
    pattern = r'([क-ह][्])([क-ह])'

    def replace_half_letter(match):
        return match.group(2)

    separated_word = re.sub(pattern, replace_half_letter, devanagari_word)
    return separated_word


# ##################################################################number conversion###################################

def katapayadi_number(shloka):
    katapayadi_map = {
        'क': 1, 'ख': 2, 'ग': 3, 'घ': 4, 'ङ': 5,
        'च': 6, 'छ': 7, 'ज': 8, 'झ': 9, 'ञ': 0,
        'ट': 1, 'ठ': 2, 'ड': 3, 'ढ': 4, 'ण': 5,
        'त': 6, 'थ': 7, 'द': 8, 'ध': 9, 'न': 0,
        'प': 1, 'फ': 2, 'ब': 3, 'भ': 4, 'म': 5,
        'य': 1, 'र': 2, 'ल': 3, 'व': 4,
        'श': 5, 'ष': 6, 'स': 7, 'ह': 8, 'ॹ': 8, 'ळ': 9,
        'अ': None, 'आ': None, 'इ': None, 'ई': None, 'उ': None, 'ऊ': None, 'ऋ': None, 'ए': None,
        'ऐ': None, 'ओ': None, 'औ': None, 'अं': None, 'अः': None,
        'ा': None, 'ि': None, 'ी': None, 'ु': None, 'ू': None, 'ृ': None, 'े': None, 'ै': None, 'ो': None, 'ौ': None,
        'ं': None, 'ँ': None, 'ऱ': 2, 'र्': 2
    }
    filtered_dict = {k: vs for k, vs in katapayadi_map.items() if vs not in ('', None)}
    number_representation = []
    nm = []
    for char in shloka:
        if char in filtered_dict:
            number_representation.append(str(filtered_dict[char]))
            nm.append(char)
    pq = number_representation
    res = ''
    for s in pq:
        res += s
    res = res.strip()
    return res

# ######################################equations#####################################################################################


x = sp.symbols('x')


def create_polynomial(coefficients):
    p = symbols('x')
    polynomial = sum(float(coeff)*p**(len(coefficients) - 1 - i) for i, coeff in enumerate(coefficients)if coeff != 0)
    return polynomial


def generate_polynomials(matrixss):
    return [create_polynomial(rows) for rows in matrixss]


def simplify_and_expand(polynomial, max_degree, rows):
    p = symbols('x')
    simplified_poly = Poly(polynomial, p).as_expr().expand().as_poly()
    coeffs = simplified_poly.all_coeffs()
    linear_poly = coeffs[0] * p + coeffs[1]
    expanded_polynomials = []
    for degree in range(2, max_degree):
        expanded_poly = str(expand(linear_poly ** degree))
        expanded_poly = expanded_poly.replace('**', '^')
        expanded_polynomials.append(f"{expanded_poly} = 0, degree = {degree+1}")
        expanded_polynomials.append(f"row=={rows}")
        expanded_polynomials.append("/.............................................../")
    return expanded_polynomials


def create_polynomialss(coefficients):
    xps = sp.symbols('x')
    degree = len(coefficients) - 1
    polynomial = sum(int(coeff) * xps**(int(degree) - int(i)) for i, coeff in enumerate(coefficients) if coeff != 0)
    return polynomial


def plot_polynomial(polynomial):
    xpss = np.linspace(-10, 10, 400)
    y = [polynomial.evalf(subs={sp.symbols('x'): val}) for val in xpss]
    plt.figure(figsize=(10, 6))
    plt.plot(xpss, y, 'o', label=str(polynomial))
    plt.title('Polynomial Graph')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.grid()
    plt.legend()
    plt.show()


def matrix(num):
    global fg, x
    res, sub, k = [], [], []
    x, y, pq, hl = 0, 0, 0, 0
    k = list(num)

    for i in range(int(math.sqrt(len(num))+1) * int(math.sqrt(len(num))+1) - len(num)):
        k.append("0")
    pq = k
    x = math.sqrt(len(pq))
    fg = x
    y = x
    f = x*y
    n = int(x)
    m = int(y)

    for i in range(m):
        sub = []
        for jdx in range(n):
            sub.append(k[hl])
            hl = hl + 1
        res.append(sub)
    return res


def graph():
    x = np.arange(len(sdfg))
    print(x)
    print(sdfg)
    plt.grid()
    plt.xticks(x)
    ax = plt.gca()
    linestlye = None
    plt.plot(x, sdfg, 'o')
    plt.show()


def karma_patha_conversion(sentence):
    words = sentence.split()
    karma_patha = []
    for i in range(len(words) - 1):
        pair = f"{words[i]}-{words[i + 1]}"
        karma_patha.append(pair)
    return ''.join(karma_patha)


def generate_pattern(num_words):
    num_words = len(num_words.split())
    pattern = []
    current_line = [1, 2, 2, 1, 1, 2]
    while True:
        pattern.extend(current_line)
        if max(current_line) > num_words:
            break
        current_line = [x + 1 for x in current_line]

    return pattern


def jata_patha(sentence):
    words = sentence.split()
    num_words = len(words)
    pattern = generate_pattern(sentence)
    result = []
    for index in pattern:
        result.append(words[(index - 1) % num_words])
    return ' '.join(result)


def generate_full_pattern(sec):
    pattern = []
    current_line = [1, 2, 2, 1, 1, 2, 3, 3, 2, 1, 1, 2, 3]
    while True:
        pattern.extend(current_line)
        if max(current_line) >= len(sec.split()):
            break
        current_line = [x + 1 for x in current_line]
    return pattern


def ghana_patha(sentence):
    words = sentence.split()
    num_words = len(words)
    pattern = generate_full_pattern(str(num_words))
    result = []
    for index in pattern:
        result.append(words[(index - 1) % num_words])
    return ' '.join(result)


def parse_polynomial(poly_str):
    # Normalize the string and split into valid terms using regex
    terms = re.findall(r'[+-]?(\d*\*?x(\^\d+)?)|[+-]?\d+', poly_str)
    coefficients = {}

    for term in terms:
        term = ''.join(term).strip()  # Join tuples and strip spaces
        if term == '':
            continue

        if 'x' in term:  # If the term contains 'x'
            if '**' in term:  # Term with power
                coeff, power = term.split('*x**')
                power = int(power)
            else:  # Term is like '5*x' or 'x'
                coeff = term[:-2]  # Everything before 'x'
                power = 1

            # Determine the coefficient value
            if coeff == '' or coeff == '+':
                coeff_value = 1
            elif coeff == '-':
                coeff_value = -1
            else:
                coeff_value = int(coeff[:-1]) if coeff.endswith('*') else int(coeff)
        else:  # It's a constant term
            power = 0
            coeff_value = int(term)

        # Update the coefficients dictionary
        coefficients[power] = coefficients.get(power, 0) + coeff_value

    # Create a list of coefficients in descending order of powers
    max_power = max(coefficients.keys())
    poly_coeffs = [coefficients.get(i, 0) for i in range(max_power, -1, -1)]

    return np.poly1d(poly_coeffs)


def find_roots(poly):
    return np.roots(poly)


def plot_roots(roots):
    plt.figure(figsize=(10, 10))
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.grid(True, which='both', linestyle='--', lw=0.5)

    # Plot each root in the complex plane
    for root in roots:
        plt.plot(root.real, root.imag, 'o', markersize=8, label=f'Root: {root:.2f}')
        plt.text(root.real, root.imag, f' {root:.2f}', fontsize=12)

    plt.title('Roots of the Polynomial in the Complex Plane')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')

    # Adjust limits based on root values
    plt.xlim(np.min(roots.real) - 1, np.max(roots.real) + 1)
    plt.ylim(np.min(roots.imag) - 1, np.max(roots.imag) + 1)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

#######################################################end############################################################################
#######################################################result#########################################################################

vedic_shloka = remove(n)
numbers = katapayadi_number(vedic_shloka)
matrixs = matrix(numbers)
result = pyfiglet.figlet_format(f"PADA PATHA: {numbers}", font = "digital" )
print(BColors.OKGREEN + result)
print()
result = pyfiglet.figlet_format(f"KARMAPATHA:{katapayadi_number(remove(karma_patha_conversion(n)))}", font = "digital", width = 300 )
print(Fore.LIGHTMAGENTA_EX + result)
print()
print(Fore.CYAN + "EQUATIONS")
print()
gfjhd = generate_polynomials(matrixs)
gfdhsj = []
fg = int(fg)
print(f"{BColors.OKCYAN}WORKING", end =" ")
print(" /", end =" ")
for row in range(fg):
    print(".", end=" ")
    simplify_and_expand(gfjhd, v , row)
    print(".", end=" ")
x = symbols('x')
print(" /", end =" ")
print()
print("")
print(f"{BColors.OKGREEN}{create_polynomialss(list(numbers))} = 0 PADA PATHA")
print("")
print(f"{BColors.OKBLUE}{create_polynomialss(list(katapayadi_number(remove(karma_patha_conversion(n)))))} = 0 KARMA PATHA")
print("")


####गोपीभाग्य मधुव्रातः श्रुंगशोदधि संधिगः | खलजीवितखाताव गलहाला रसंधरः ||