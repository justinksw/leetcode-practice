
- [88. Merge Sorted Array](#88-merge-sorted-array)
- [27. Remove Element](#27-remove-element)
- [26. Remove Duplicates from Sorted Array](#26-remove-duplicates-from-sorted-array)
- [80. Remove Duplicates from Sorted Array II](#80-remove-duplicates-from-sorted-array-ii)
- [169. Majority Element](#169-majority-element)
- [189. Rotate Array](#189-rotate-array)
- [121. Best Time to Buy and Sell Stock](#121-best-time-to-buy-and-sell-stock)
- [122. Best Time to Buy and Sell Stock II](#122-best-time-to-buy-and-sell-stock-ii)
- [55. Jump Game](#55-jump-game)
- [45. Jump Game II](#45-jump-game-ii)
- [274. H-Index](#274-h-index)
- [380. Insert Delete GetRandom O(1)](#380-insert-delete-getrandom-o1)
- [238. Product of Array Except Self](#238-product-of-array-except-self)
- [134. Gas Station](#134-gas-station)
- [13. Roman to Integer](#13-roman-to-integer)
- [12. Integer to Roman](#12-integer-to-roman)
- [58. Length of Last Word](#58-length-of-last-word)
- [14. Longest Common Prefix](#14-longest-common-prefix)


# 88. Merge Sorted Array

```
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        
        midx = m - 1
        nidx = n - 1

        right = m + n - 1

        # nums1 = [1, 2, 3, 0, 0, 0]
        # nums2 = [2, 5, 6]

        while nidx >= 0:
            
            if midx >= 0 and nums1[midx] > nums2[nidx]:
                
                nums1[right] = nums1[midx]

                midx -= 1

            else:
                nums1[right] = nums2[nidx]

                nidx -= 1

            right -= 1
```

# 27. Remove Element

```
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        
        k = 0

        for i in range(len(nums)):

            if nums[i] != val:
                
                nums[k] = nums[i]

                k += 1

        return k
```

# 26. Remove Duplicates from Sorted Array

```
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        k = 1
        
        for i in range(1, len(nums)):

            if nums[i] != nums[i - 1]:

                nums[k] = nums[i]

                k += 1

        return k
```

# 80. Remove Duplicates from Sorted Array II

```
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        k = 2

        for i in range(2, len(nums)):

            if nums[i] != nums[k - 2]:
                
                nums[k] = nums[i]
        
                k += 1

        return k
```

# 169. Majority Element

** Hash map **

```
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        a = {}

        count = 0
        majority = 0

        for num in nums:

            a[num] = a.get(num, 0) + 1

            if a[num] > count:
                
                majority = num
                count = a[num]

        return majority
```

# 189. Rotate Array

```
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """

        k = k % len(nums)

        nums[:] = nums[-k:] + nums[:-k]
```

# 121. Best Time to Buy and Sell Stock

```
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        
        buy_price = prices[0]

        profit = 0

        for price in prices:
            
            if price - buy_price < 0:
                
                buy_price = price

            else:
                if price - buy_price > profit:
                    profit = price - buy_price

        return profit
```

# 122. Best Time to Buy and Sell Stock II

```
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """

        profit = 0

        for i in range(1, len(prices)):

            if prices[i] > prices[i - 1]:

                profit += (prices[i] - prices[i - 1])

        return profit
```

# 55. Jump Game

```
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        
        goal = len(nums) - 1
        
        for i in range(len(nums)-2, -1, -1):
            
            if nums[i] + i >= goal:  # This line

                goal = i

        if goal == 0:
            return True
        
        else:
            return False
```

# 45. Jump Game II

```
class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        n = len(nums) - 1
        
        jumps = 0
        i_lower = 0
        i_upper = 0

        while i_upper < n:

            farthest = 0

            for i in range(i_lower, i_upper + 1):

                farthest = max(farthest, i + nums[i])

            i_lower = i_upper + 1
            i_upper = farthest

            jumps += 1

        return jumps
```

# 274. H-Index

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

# 238. Product of Array Except Self

```
class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """

        n = len(nums)

        result = [1 for _ in range(n)]

        to_right = 1

        for i in range(n):
            result[i] *= to_right
            to_right *= nums[i]

        to_left = 1

        for i in range(n-1, -1, -1):
            result[i] *= to_left
            to_left *= nums[i]

        return result
```

# 134. Gas Station

```
class Solution(object):
    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        if sum(gas) < sum(cost):
            return -1
                
        curernt_gas = 0
        start = 0
        for i in range(len(gas)):
            curernt_gas += gas[i] - cost[i]
            if curernt_gas < 0:
                curernt_gas = 0
                start = i + 1

        return start
```

# 13. Roman to Integer

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

# 12. Integer to Roman

重點在於`value_symbols`

```
class Solution(object):
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        
        value_symbols = [
            (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
            (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
            (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'),
            (1, 'I'),
        ]
        
        res = []

        # 3749 // 1000 = 3 --> MMM
        # 749 // 500 = 1 --> D
        # 249 // 100 = 2 --> CC
        # 49 // 40 = 1 --> XL
        # 9 // 9 = 1 --> IX
        # 3749 --> MMMDCCXLIX

        for value, symbol in value_symbols:
            if num == 0:
                break
            count = num // value
            res.append(symbol * count)
            num -= count * value

        return ''.join(res)  
```

# 58. Length of Last Word

```
class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        s = s.rstrip(" ").split(" ")
        return len(s[-1])
```

# 14. Longest Common Prefix

```
class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        
        prefix = strs[0]
        prefix_len = len(prefix)

        for s in strs[1:]:
            
            while s[:prefix_len] != prefix[:prefix_len]:
                    
                prefix_len -= 1

                if prefix_len == 0:
                    return ""

        return prefix[:prefix_len]
```




