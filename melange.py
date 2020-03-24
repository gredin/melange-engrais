import numpy as np
from scipy.optimize import nnls

engrais_npk = [
    (14, 0, 0),
    (2, 8, 6),
    (3, 4, 10),
]

dosage_npk = (75, 100, 50)  # U = kg/hectare

engrais_npk_grams = [map(lambda x: x / 100, npk) for npk in engrais_npk]
dosage_npk_grams_by_squaremeter = map(lambda x: x * 1000 / (100 * 100), dosage_npk)

A = np.array([[n, p, k] for n, p, k in engrais_npk_grams])
b = np.array(list(dosage_npk_grams_by_squaremeter))

# non-negative least squares solver
solution = nnls(A, b)[0]

print("engrais disponibles :")
for npk in engrais_npk:
    print("- engrais %d - %d - %d" % npk)

print("")

print("dosage visé : %d - %d - %d U (kg/ha)" % dosage_npk)
print("")
print("---")
print("")

dosage_approche = np.matmul(A, np.array([solution]).T) * (100 * 100) / 1000

print("dosage approché : %.1f - %.1f - %1.f U (kg/ha)" % tuple([x for x in dosage_approche.T[0]]))

print("")

print("quantité à utiliser par mètre carré :")

for i, coeff in enumerate(solution):
    print("- %.1f g de l'engrais %d - %d - %d" % ((coeff,) + engrais_npk[i]))
