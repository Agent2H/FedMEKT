#
import numpy as np
# num_clients_A = 30
# num_clients_B = 0
# num_clients_AB = 30
# modalities = ["A" for _ in range(num_clients_A)] + ["B" for _ in range(
#     num_clients_B)] + ["AB" for _ in range(num_clients_AB)]
#
# print(modalities)
fall_test = np.random.choice(range(1, 31), 3, replace=False)
adl_test = np.random.choice(range(1, 41), 4, replace=False)

fall_public = np.random.choice([i for i in range(1, 31) if i not in fall_test], 3, replace=False)
adl_public = np.random.choice([i for i in range(1, 41) if i not in adl_test], 4, replace=False)
print("fall_public", fall_public)
print("fall_test", fall_test)
print("adl_public", adl_public)
print("adl_test", adl_test)