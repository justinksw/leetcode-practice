
- [12. Integer to Roman](#12-integer-to-roman)
- [13. Roman to Integer](#13-roman-to-integer)
- [274. H-Index](#274-h-index)
- [380. Insert Delete GetRandom O(1)](#380-insert-delete-getrandom-o1)


# 12. Integer to Roman

https://leetcode.com/problems/integer-to-roman/description/?envType=study-plan-v2&envId=top-interview-150

學習一下簡潔的寫法。

```
class Solution(object):
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        
        value_symbols = [
            (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
            (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'), (10, 'X'),
            (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
        ]
        
        res = []

        for value, symbol in value_symbols:
            if num == 0:
                break
            count = num // value
            res.append(symbol * count)
            num -= count * value

        return ''.join(res)  
```

# 13. Roman to Integer

https://leetcode.com/problems/roman-to-integer/description/?envType=study-plan-v2&envId=top-interview-150

```
class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """

        res = 0
        
        roman = {
            "I": 1,
            "V": 5,
            "X": 10,
            "L": 50,
            "C": 100,
            "D": 500,
            "M": 1000,
        }

        # e.g. s = "LVIII"
        # print:
        # L, V
        # V, I
        # I, I
        # I, I

        for a, b in zip(s, s[1:]):
            
            if roman[a] < roman[b]:
                res -= roman[a]
            else:
                res += roman[a]

        return res + roman[s[-1]]

```

# 274. H-Index

https://leetcode.com/problems/h-index/?envType=study-plan-v2&envId=top-interview-150

The **h-index** is defined as the maximum value of `h` such that the given researcher has published at least `h` papers that have each been cited at least `h` times.

這一題確實有點難理解。

```
class Solution(object):
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """

        n = len(citations)
        citations.sort()

        for i,v in enumerate(citations):
            if n - i <= v:
                return n - i
        return 0
```

# 380. Insert Delete GetRandom O(1)

https://leetcode.com/problems/insert-delete-getrandom-o1/description/?envType=study-plan-v2&envId=top-interview-150

```
class RandomizedSet(object):

    def __init__(self):
        
        self.lst = []
        self.idx_map = {}

    def is_exist(self, val):
        return val in self.idx_map

    def insert(self, val):
        """
        :type val: int
        :rtype: bool
        """
        if self.is_exist(val):
            return False
        
        self.lst.append(val)
        self.idx_map[val] = len(self.lst) - 1
        return True

    def remove(self, val):
        """
        :type val: int
        :rtype: bool
        """
        if not self.is_exist(val):
            return False
        
        idx = self.idx_map[val]
       
        # 在需要被remove的item的位置上複製最後一個位置的值
        self.lst[idx] = self.lst[-1]
        # 同時記下更改位置後 原本最後一個值的位置
        self.idx_map[self.lst[-1]] = idx

        # pop走最後一個值
        self.lst.pop()
        del self.idx_map[val]
        return True

    def getRandom(self):
        """
        :rtype: int
        """
        import random
        return random.choice(self.lst)
        
# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()
```

