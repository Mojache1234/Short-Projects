# def is_leap(year: int):
#     return year % 400 == 0 if year % 100 == 0 else year % 4 == 0

is_leap = lambda year: year % 400 == 0 if year % 100 == 0 else year % 4 == 0

while True:
    leap = is_leap(int(input()))
    print(leap)