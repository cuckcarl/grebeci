
ids = [7, 9, 10, 13]


result = {}
for id in ids:
	with open('./predict_result/result.epoch_%d' % id) as f:
		for line in f:
			arr = line.strip().split(',')
			if arr[0] not in result:
				result[arr[0]] = []
			result[arr[0]].append(float(arr[1]))

outf = open('result.epoch_merge', 'w')
outf.write('id,is_iceberg\n')

for id in result:
	outf.write('%s,%f\n' % (id, sum(result[id]) / len(ids)))