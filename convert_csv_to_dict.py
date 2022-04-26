import csv

file = open('class_dict.csv')
type(file)
csvreader = csv.reader(file)
header = next(csvreader)
print(header)
rows = {}
for row in csvreader:
    rows.update({int(row[0]): row[1]})
print(rows)
file.close()