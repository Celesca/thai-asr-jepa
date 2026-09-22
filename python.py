from math import comb

total = 0
for k in range(4, 9):
    ways = comb(40, k)
    print(k, ways)
    total += ways
print("Total plans:", total)