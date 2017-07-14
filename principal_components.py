import os
import sys
import math
import array
import numpy
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vectors', required=True)
    parser.add_argument('--words', type=file)
    parser.add_argument('--dim', type=int, required=True)
    parser.add_argument('--norm', action="store_true")
    parser.add_argument('--top', type=int, default=10)
    parser.add_argument('--head', type=int)
    parser.add_argument('--new-dim', type=int, default=0)
    return parser.parse_args()

def read_binary_matrix(fname, dim):
    d1 = dim
    d2 = os.path.getsize(fname) // dim // 4
    sys.stderr.write('%d words\n' % (d2, ))
    with open(fname, 'rb') as f:
        return numpy.fromfile(f, dtype='float32', count=d1 * d2).reshape((d2, d1))

def get_words(words_file):
    words_file.readline()  # the first line contains metainformation about dimension and vocabulary size
    words = []
    for word in words_file:
        word = word.strip()
        words.append(word)
    return words

def scalar(v, w):
    return v.transpose().dot(w)

def cos(v, w):
    return scalar(v, w) / math.sqrt(scalar(v, v)) / math.sqrt(scalar(w, w))

def normalize(W):
    for i in xrange(W.shape[0]):
        norm = math.sqrt(sum(elem**2 for elem in W[i]))
        if norm != 0.0:
            W[i] /= norm

def print_clusters(selected, W, j):
    THRESHOLD = 0.6
    source = [(idx, word) for (val, word, idx) in selected]
    source.sort()
    clusters = []
    for idx, word in source:
        w = numpy.array(W[idx])
        word = word + "_" + str(w[j])
        appended = False
        for i in xrange(len(clusters)):
            v = clusters[i][0]
            if cos(v, w) > THRESHOLD:
                clusters[i][0] += w
                clusters[i][1].append(word)
                appended = True
                break
        if not appended:
            clusters.append([w, [word]])
    for cluster in clusters:
        print "\t" + " ".join(cluster[1])


if __name__ == "__main__":
    args = get_args()
    words = get_words(args.words)
    W = read_binary_matrix(args.vectors, args.dim)
    if args.head:
        W = W[:args.head,:]
    N, d = W.shape

    if args.norm:
        normalize(W)

    U, Sigma, Vt = numpy.linalg.svd(W, full_matrices=0)
    W = W.dot(Vt.transpose())

    if args.new_dim != 0:
        d = args.new_dim
        W = W[:,:d]

    if args.norm:
        normalize(W)

    for j in xrange(d):
        print j, Sigma[j], "========="
        ordered = [(W[i,j], words[i], i) for i in xrange(N) if abs(W[i,j]) > 0.0]
        ordered.sort()
        print_clusters([z for z in ordered if z[0] < 0][:args.top], W, j)
        print "\n-----------"
        print_clusters([z for z in ordered if z[0] > 0][-args.top:], W, j)
        print ""


