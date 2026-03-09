import string

def validate_password(passw):
    min_length = 8


    # any() is used for all elements what if it is in list,tuple or dic. 
    # Using any() we use for long in side
    if len(passw) < min_length:
        return "Password must be at least 8 characters long"

    if not any(char in string.ascii_uppercase for char in passw):
        return "Password must contain at least one uppercase letter"

    if not any(char in string.ascii_lowercase for char in passw):
        return "Password must contain at least one lowercase letter"

    if not any(char in string.digits for char in passw):
        return "Password must contain at least one digit"

    if not any(char in string.punctuation for char in passw):
        return "Password must contain at least one special character"

    return "Password is strong ✅"


passw = input("Enter password: ")
validate_password(passw)