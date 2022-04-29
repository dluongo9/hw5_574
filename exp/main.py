import numpy as np
test_array = np.array([[.9, .1, 0], [.2, .7, .1]])

random_gen = np.random.default_rng()
#selection = random_gen.choice(test_array, axis=1, p=test_array)
print(test_array)
#print(selection)
print([np.random.choice(len(row), p=row) for row in test_array])
