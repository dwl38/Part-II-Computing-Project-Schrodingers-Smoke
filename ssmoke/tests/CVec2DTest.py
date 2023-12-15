import numpy as np
import time
from ..common import *
from ..cvec2d import CVec2D

#====================================================================================================
### DEPRECATED: The CVec2D class has been deprecated.
#====================================================================================================
# Performance tests for the CVec2D class.
#====================================================================================================

# Predefinitions

rng = np.random.default_rng()

def rndcmplx(offset=0,scale=10):
    return offset + complex(rng.normal(0, scale), rng.normal(0, scale))

def testeq(left, right):
    if left == right:
        return 'Passed'
    diff = left - right
    if isscalar(diff):
        return 'Failed with delta ' + str(round_sig(diff))
    return 'Failed with delta ' + str(diff)

print()
print('=' * 100)
print(' Performance tests for the CVec2D class.')
print('=' * 100)
print()

#----------------------------------------------------------------------------------------------------
# Arithmetic tests

print('Performing arithmetic tests...')
print()

a1 = rndcmplx()
a2 = rndcmplx()
b1 = rndcmplx()
b2 = rndcmplx()
testA = CVec2D(a1, a2)
testB = CVec2D(b1, b2)
testC = testA + testB
passed = ((testC[0].real == (testA[0].real + testB[0].real)) and (testC[0].imag == (testA[0].imag +
          testB[0].imag)) and (testC[1].real == (testA[1].real + testB[1].real)) and (testC[1].imag
          == (testA[1].imag + testB[1].imag)))
print(f'Check: {testA} + {testB} = {testC} ' + ('Passed' if passed else 'Failed'))
testD = testB + testA
print(r'    Check: A + B = B + A ' + testeq(testC, testD))
print()

a1 = rndcmplx()
a2 = rndcmplx()
b1 = rndcmplx()
b2 = rndcmplx()
testA = CVec2D(a1, a2)
testB = CVec2D(b1, b2)
testC = testA - testB
passed = ((testC[0].real == (testA[0].real - testB[0].real)) and (testC[0].imag == (testA[0].imag -
          testB[0].imag)) and (testC[1].real == (testA[1].real - testB[1].real)) and (testC[1].imag
          == (testA[1].imag - testB[1].imag)))
print(f'Check: {testA} - {testB} = {testC} ' + ('Passed' if passed else 'Failed'))
testD = (testA + testB) - testB
print(r'    Check: (A + B) - B = A ' + testeq(testA, testD))
testD = testA + (-testB)
print(r'    Check: A - B = A + (-B) ' + testeq(testC, testD))
testD = testB - testA
print(r'    Check: A - B = - (B - A) ' + testeq(testC, -testD))
print()

a1 = rndcmplx()
a2 = rndcmplx()
b1 = rndcmplx()
b2 = rndcmplx()
s = rndcmplx()
testA = CVec2D(a1, a2)
testB = CVec2D(b1, b2)
testC = s * testA
passed = ((testC[0] == s * testA[0]) and (testC[1] == s * testA[1]))
print(f'Check: {round_sig(s)} * {testA} = {testC} ' + ('Passed' if passed else 'Failed'))
testD = testA * s
print(r'    Check: s * A = A * s ' + testeq(testC, testD))
testC = s * (testA + testB)
testD = (s * testA) + (s * testB)
print(r'    Check: s * (A + B) = sA + sB ' + testeq(testC, testD))
print()

a1 = rndcmplx()
a2 = rndcmplx()
s = rndcmplx()
testA = CVec2D(a1, a2)
testC = testA / s
passed = ((testC[0] == testA[0] / s) and (testC[1] == testA[1] / s))
print(f'Check: {testA} / {round_sig(s)} = {testC} ' + ('Passed' if passed else 'Failed'))
testD = (1/s) * testA
print(r'    Check: A / s = (1/s) * A ' + testeq(testC, testD))
print()

a1 = rndcmplx()
a2 = rndcmplx()
b1 = rndcmplx()
b2 = rndcmplx()
testA = CVec2D(a1, a2)
testB = CVec2D(b1, b2)
testC = complex(testA * testB)
passed = (testC == ((testA[0].conjugate() * testB[0]) + (testA[1].conjugate() * testB[1])))
print(f'Check: <{testA}|{testB}> = {round_sig(testC)} ' + ('Passed' if passed else 'Failed'))
testD = complex(testB * testA)
print(r'    Check: <A|B> = <B|A>* ' + testeq(testC, testD.conjugate()))
x1 = rndcmplx()
x2 = rndcmplx()
testX = CVec2D(x1, x2)
testC = (testA + testB) * testX
testD = (testA * testX) + (testB * testX)
print(r'    Check: <A + B|X> = <A|X> + <B|X> ' + testeq(testC, testD))
x1 = rndcmplx()
x2 = rndcmplx()
testX = CVec2D(x1, x2)
testC = testX * (testA + testB)
testD = (testX * testA) + (testX * testB)
print(r'    Check: <X|A + B> = <X|A> + <X|B> ' + testeq(testC, testD))
s = rndcmplx()
testC = testA * (s * testB)
testD = s * (testA * testB)
print(r'    Check: <A|sB> = s<A|B> ' + testeq(testC, testD))
s = rndcmplx()
testC = (s * testA) * testB
testD = s.conjugate() * (testA * testB)
print(r'    Check: <sA|B> = (s*)<A|B> ' + testeq(testC, testD))
testC = testA * testA
print(r'    Check: <A|A> = |A|^2 ' + testeq(testC, testA.mag_sq()))
print()

print('-' * 100)
print()

#----------------------------------------------------------------------------------------------------
# Timing tests

print('Performing timing tests...')
print()

n_trials = 1000000
a1 = rndcmplx()
a2 = rndcmplx()
b1 = rndcmplx()
b2 = rndcmplx()
testA = CVec2D(a1, a2)
testB = CVec2D(b1, b2)
start = time.time()
for _ in range(n_trials):
    testC = testA + testB
end = time.time()
print(f'Operation (A + B) with CVec2D:        {round_sig(1000 * (end - start) / n_trials)} ms per operation')

testA = np.array((a1, a2));
testB = np.array((b1, b2));
start = time.time()
for _ in range(n_trials):
    testC = testA + testB
end = time.time() 
print(f'Operation (A + B) with numpy.ndarray: {round_sig(1000 * (end - start) / n_trials)} ms per operation')
print()

a1 = rndcmplx()
a2 = rndcmplx()
b1 = rndcmplx()
b2 = rndcmplx()
testA = CVec2D(a1, a2)
testB = CVec2D(b1, b2)
start = time.time()
for _ in range(n_trials):
    testC = testA * testB
end = time.time()
print(f'Operation < A | B > with CVec2D:        {round_sig(1000 * (end - start) / n_trials)} ms per operation')

testA = np.array((a1, a2));
testB = np.array((b1, b2));
start = time.time()
for _ in range(n_trials):
    testC = np.vdot(testA, testB)
end = time.time() 
print(f'Operation < A | B > with numpy.ndarray: {round_sig(1000 * (end - start) / n_trials)} ms per operation')
print()

n_list = 1000
n_trials = 1000
testA = [CVec2D(rndcmplx(), rndcmplx()) for _ in range(n_list)]
testB = [CVec2D(rndcmplx(), rndcmplx()) for _ in range(n_list)]
start = time.time()
for _ in range(n_trials):
    testC = []
    for i in range(n_list):
        testC.append(testA[i] + testB[i])
end = time.time()
print(f'Broadcasting (A + B) over list of {n_list} CVec2Ds:     {round_sig(1000 * (end - start) / n_trials)} ms per operation')
testA = np.array(testA, dtype=object)
testB = np.array(testB, dtype=object)
start = time.time()
for _ in range(n_trials):
    testC = testA + testB
end = time.time()
print(f'Broadcasting (A + B) as ndarray of {n_list} CVec2Ds:    {round_sig(1000 * (end - start) / n_trials)} ms per operation')
testA = np.empty((n_list, 2), dtype=complex)
testB = np.empty((n_list, 2), dtype=complex)
for i in range(n_list):
    testA[i][0] = rndcmplx()
    testA[i][1] = rndcmplx()
    testB[i][0] = rndcmplx()
    testB[i][1] = rndcmplx()
start = time.time()
for _ in range(n_trials):
    testC = testA + testB
end = time.time()
print(f'Broadcasting (A + B) as ndarray of shape ({n_list}, 2): {round_sig(1000 * (end - start) / n_trials)} ms per operation')
print()

testA = [CVec2D(rndcmplx(), rndcmplx()) for _ in range(n_list)]
testB = [CVec2D(rndcmplx(), rndcmplx()) for _ in range(n_list)]
start = time.time()
for _ in range(n_trials):
    testC = []
    for i in range(n_list):
        testC.append(testA[i] * testB[i])
end = time.time()
print(f'Broadcasting <A|B> over list of {n_list} CVec2Ds:     {round_sig(1000 * (end - start) / n_trials)} ms per operation')
testA = np.array(testA, dtype=object)
testB = np.array(testB, dtype=object)
start = time.time()
for _ in range(n_trials):
    testC = testA * testB
end = time.time()
print(f'Broadcasting <A|B> as ndarray of {n_list} CVec2Ds:    {round_sig(1000 * (end - start) / n_trials)} ms per operation')
testA = np.empty((n_list, 2), dtype=complex)
testB = np.empty((n_list, 2), dtype=complex)
for i in range(n_list):
    testA[i][0] = rndcmplx()
    testA[i][1] = rndcmplx()
    testB[i][0] = rndcmplx()
    testB[i][1] = rndcmplx()
start = time.time()
for _ in range(n_trials):
    testC = np.einsum('ij,ij->i', np.conj(testA), testB)
end = time.time()
print(f'Broadcasting <A|B> as ndarray of shape ({n_list}, 2): {round_sig(1000 * (end - start) / n_trials)} ms per operation')
print()

print()
print()
input(r'Press Enter to quit...')


