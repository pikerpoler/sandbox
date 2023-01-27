
def combinations(n, r):
    # this is definitely not an adjustment of itertools.combinations
    pool = tuple(range(1, n))
    n = n-1
    if r > n:
        return
    indices = list(range(r))
    res = [tuple(pool[i] for i in indices)]
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return res
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        res += [tuple(pool[i] for i in indices)]


def all_ways_to_cut(word, cuts):
    # indexes = combinations(range(1, len(word)), cuts -1)
    indexes = combinations(len(word), cuts-1)
    res = []

    for index_list in indexes:
        word_list = []
        for left, right in zip([0] + [j for j in index_list], [j for j in index_list] + [len(word)]):
            word_list += [word[left:right]]
        res.append(word_list)
    return res


all_ways = all_ways_to_cut("doggy", 0)
print(sorted(all_ways))
