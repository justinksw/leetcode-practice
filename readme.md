[**Leetcode Interview Questions 150**](https://leetcode.com/studyplan/top-interview-150/)


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
- [135. Candy (Hard)](#135-candy-hard)
- [13. Roman to Integer](#13-roman-to-integer)
- [12. Integer to Roman](#12-integer-to-roman)
- [58. Length of Last Word](#58-length-of-last-word)
- [14. Longest Common Prefix](#14-longest-common-prefix)
- [151. Reverse Words in a String](#151-reverse-words-in-a-string)
- [28. Find the Index of the First Occurrence in a String](#28-find-the-index-of-the-first-occurrence-in-a-string)
- [125. Valid Palindrome](#125-valid-palindrome)
- [392. Is Subsequence](#392-is-subsequence)
- [167. Two Sum II - Input Array Is Sorted](#167-two-sum-ii---input-array-is-sorted)
- [11. Container With Most Water](#11-container-with-most-water)
- [15. 3 Sum](#15-3-sum)
- [209. Minimum Size Subarray Sum](#209-minimum-size-subarray-sum)



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

有機會可以用在其他地方。當想要移除數列中的某個數值（多個 duplicates）而位置不重要的時候。

關於移除某個單個unique的數值，可以參考 [380. Insert Delete GetRandom O(1)](#380-insert-delete-getrandom-o1)

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

        # [1, 2, 3] remove 2, --> [1, 3, _] --> k = 2
        # [2, 3, 3, 2] remove 2, --> [3, 3, _, _] --> k = 2

        # _ denote the values we dont care

        return k
```

# 26. Remove Duplicates from Sorted Array

和上一題的分別是：這一題是移除重複的值。

上一題：

[2, 3, 3, 2] 移除 2 --> [3, 3, _, _]

這一題：

[1, 1, 2] --> [1, 2, _]  k = 2

[0, 0, 1, 1, 1, 2, 2, 3, 3, 4] --> [0, 1, 2, 3, 4, _, _, _, _, _] --> k = 5


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

Given an integer array nums sorted in non-decreasing order, remove some duplicates in-place such that each unique element appears **at most twice**. The relative order of the elements should be kept the same.

[1, 1, 1, 2, 2, 3] --> [1, 1, 2, 2, 3, _]

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

**Hash map**

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

# 135. Candy (Hard)

貪婪演算法（英語：greedy algorithm），又稱貪心演算法，是一種在每一步選擇中都採取在當前狀態下最好或最佳（即最有利）的選擇，從而希望導致結果是最好或最佳的演算法。比如在旅行推銷員問題中，如果旅行員每次都選擇最近的城市，那這就是一種貪婪演算法。

The Nuts and Bolts of the Two-Pass Method

```
class Solution:
    def candy(self, ratings: List[int]) -> int:
        n = len(ratings)
        candies = [1] * n 

        for i in range(1, n):
            if ratings[i] > ratings[i-1]:
                candies[i] = candies[i-1] + 1

        for i in range(n-2, -1, -1):
            if ratings[i] > ratings[i+1]:
                candies[i] = max(candies[i], candies[i+1] + 1)
        
        return sum(candies)
```

One-Pass Greedy Algorithm: Up-Down-Peak Method

```
class Solution:
    def candy(self, ratings: List[int]) -> int:
        if not ratings:
            return 0
        
        ret, up, down, peak = 1, 0, 0, 0
        
        for prev, curr in zip(ratings[:-1], ratings[1:]):
            if prev < curr:
                up, down, peak = up + 1, 0, up + 1
                ret += 1 + up
            elif prev == curr:
                up = down = peak = 0
                ret += 1
            else:
                up, down = 0, down + 1
                ret += 1 + down - int(peak >= down)
        
        return ret
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

# 151. Reverse Words in a String

Note: if " " --> False

```
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """

        s = s.strip(" ").split(" ")
        
        r = []

        for i in range(len(s)-1, -1, -1):
            if s[i]:
                if i != 0:
                    temp = s[i] + " "
                else:
                    temp = s[i]

                r.append(temp)

        return "".join(r)
```

# 28. Find the Index of the First Occurrence in a String

```
class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        needle_len = len(needle)

        flag = False

        for i in range(0, len(haystack)):
            if haystack[i: i + needle_len] == needle:
                flag = True
                break

        if flag:
            return i

        else:
            return -1 
```

# 125. Valid Palindrome

Note: Remove all non-alphanumeric characters. Alphanumeric characters include letters and numbers.

"A ma," --> "ama" is poalindrome 

```
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """

        s = s.lower()
        s = ''.join(char for char in s if char.isalnum())
        
        s_len = len(s)

        for i in range(s_len // 2):
            if s[i] != s[s_len-1-i]:
                return False

        return True
```

# 392. Is Subsequence

Need revision on this. ...this question is actually easy.

(1) While loop, two pointers

(2) return the == 

```
class Solution(object):
    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """

        sp = tp = 0

        while sp < len(s) and tp < len(t):
            if s[sp] == t[tp]:
                sp += 1
            tp += 1
        
        return sp == len(s)
```

# 167. Two Sum II - Input Array Is Sorted

The last four lines worth remember. The point is, this array is **sorted**.

Two pointers move from (1) left to right, and (2) right to left.

```
class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        
        left = 0
        right = len(numbers) - 1

        while right > left:

            total = numbers[left] + numbers[right]
            
            if total == target:
                return [left + 1, right +1]

            elif total > target:
                right -= 1

            else:
                left += 1
                
        return False
```

# 11. Container With Most Water

I use the similar logic in the previous question [167. Two Sum II - Input Array Is Sorted](#167-two-sum-ii---input-array-is-sorted)

Two pointers, left to right, right to left.

```
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        
        left = 0
        right = len(height) - 1

        max_area = 0

        while right > left:
            
            width = right - left

            if height[left] > height[right]:

                area = height[right] * width

                right -= 1
            
            else:
                area = height[left] * width

                left += 1

            if area > max_area:
                max_area = area

        return max_area
```

# 15. 3 Sum

My solution. The time complexity is O(N^2)

原本我還打算抄別人的solution, 結果一submit發現有bug ==, 想不出更好的解法之前 O(N^2) 能解決問題也只能照用。

```
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """

        nums.sort()

        res = []

        for i in range(len(nums)):
            
            j = i + 1
            k = len(nums) - 1

            while j < k:
                
                total = nums[i] + nums[j] + nums[k]

                if total == 0:
                    ans = [nums[i], nums[j], nums[k]]
                    if ans not in res:
                        res.append(ans)
                    j += 1
                
                elif total > 0:
                    k -= 1

                elif total < 0:
                    j += 1

        return res
```

# 209. Minimum Size Subarray Sum

題目在於找出最小的長度的滑動窗口，其中的所有數值總和sum等於target。而不是找出任意間隔最小的兩個數字。

這個解法的time complexity是 O(N)

```
class Solution(object):
    def minSubArrayLen(self, target, nums):
        """
        :type target: int
        :type nums: List[int]
        :rtype: int
        """
        min_len = float("inf")
        left = 0
        cur_sum = 0

        for right in range(len(nums)):
            cur_sum += nums[right]

            while cur_sum >= target:
                if right - left + 1 < min_len:
                    min_len = right - left + 1
                cur_sum -= nums[left]
                left += 1
        
        return min_len if min_len != float("inf") else 0
```