from functools import reduce

n = int(input())
student_marks = {}
for _ in range(n):
    name, *line = input().split()
    scores = list(map(float, line))
    student_marks[name] = scores

query_name = input()
print(
    '{:0.2f}'.format(reduce(
        lambda x, y: x + y,
        student_marks[query_name]) / len(student_marks[query_name]
    ))
)