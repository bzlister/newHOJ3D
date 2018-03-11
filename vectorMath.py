import math

def getMagnitude(vec):
    return math.sqrt(vec[0]*vec[0]+ vec[1]*vec[1] + vec[2]*vec[2])

def dotProduct(vec1, vec2):
    return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]

def scalarMult(s, vec):
    return [s*vec[0], s*vec[1], s*vec[2]]