

class Solution:
    def isvalid(self,s : str ) -> bool:
        correct = []

        for char in s:
            if char == '[' or char == '{' or char == '(':
                correct.append(char)
            
            else:
                if not correct:
                    return False
                top = correct.pop()

                if char == ')' and top != '(':
                    return False
                if char == '}' and top != '{':
                    return False
                if char == ']' and top != '[':
                    return False
        return len(correct) == 0
    
str = Solution()
print(str.isvalid('(())'))



