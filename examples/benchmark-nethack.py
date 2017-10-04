#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from collections import defaultdict
import csv
import datetime
import fileinput
import hashlib
import itertools
import sys
import time
import uuid

import traildb


def to_dict(row):
	return dict(i.split('=', 1) for i in row)


def parse_lines(inp):
	lines = csv.reader(inp, delimiter=":")
	return itertools.imap(to_dict, lines)


def generate(inp, name):
	lines = parse_lines(inp)
	first = next(lines)

	fields = sorted(first.keys())
	tdb_cons = traildb.TrailDBConstructor(name, fields)

	for row in itertools.chain([first], lines):
		values = tuple([row[k] for k in fields])
		assert len(values) == len(fields)
		tdb_cons.add(
			hashlib.md5(row['name']).hexdigest(),
			datetime.datetime.strptime(row['deathdate'], "%Y%m%d"),
			values,
		)

	tdb_cons.finalize()


def benchmark(tdb):
	result_sum = defaultdict(int)
	result_n = defaultdict(int)

	for uuid, cur in tdb.trails():
		for event in cur:
			death_type = event.death.split(' ')[0].lower()
			result_sum[death_type] += int(event.points)
			result_n[death_type] += 1

	return {k: (result_sum[k] / result_n[k], result_n[k]) for k in result_sum}


def main(args):
	if args.generate:
		generate(fileinput.input(files=args.files if len(args.files) > 0 else ('-', )), args.name)

	if args.benchmark:
		tdb = traildb.TrailDB("{name}.tdb".format(name=args.name))
		start = time.time()
		results = benchmark(tdb)
		print(results)
		print("Timing: {}".format(time.time() - start))


def parse_args(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--generate",
		action="store_true",
		help="Regenerate the traildb. Pass input filename as an arg or pipe input from stdin")
	parser.add_argument(
		"--benchmark",
		action="store_true",
		help="Run the benchmark test")
	parser.add_argument(
		"--name",
		default="nethack",
		help="Name of the traildb file (Without .tdb)")
	parser.add_argument(
		'files',
		metavar='FILE',
		nargs='*',
		help="Input files")

	args = parser.parse_args(argv)
	return args

if __name__ == '__main__':
	args = parse_args(sys.argv[1:])
	main(args)
