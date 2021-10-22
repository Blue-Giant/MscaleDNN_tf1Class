import tensorflow as tf
import numpy as np


def random_boundary():
    ul = lambda x: 0.0
    ur = lambda x: 0.0
    return ul, ur


def rangdom_ceof1(x=None, xi1=None, xi2=None, K=5, alpha=0.8):
    a = 1.0
    sum = 0.0
    for k in range(K):
        xi1k = xi1[k]
        xi2k = xi2[k]
        temp = tf.pow(float(k+1), -1.0*alpha)*(xi1k*tf.sin(k*x)+xi2k*tf.cos(k*x))
        sum = sum + temp
    a = a+0.5*tf.sin(sum)
    return a


def rangdom_force1(x=None, xi1=None, xi2=None, K=5, alpha=0.8):
    a = 1.0
    sum = 0.0
    for k in range(K):
        xi1k = xi1[k]
        xi2k = xi2[k]
        temp = tf.pow(float(k+1), -1.0*alpha)*(xi1k*tf.sin(k*x)+xi2k*tf.cos(k*x))
        sum = sum + temp
    return sum


def rangdom_diff_force2x_1(x=None, xi1=None, xi2=None, K=5, alpha=0.8):
    a = 1.0
    sum = 0.0
    for k in range(K):
        xi1k = xi1[k]
        xi2k = xi2[k]
        temp = tf.pow(float(k+1), -1.0*alpha)*(k*xi1k*tf.cos(k*x)-k*xi2k*tf.sin(k*x))
        sum = sum + temp
    return sum


def rangdom_exact_solution_1(x=None, xi1=None, xi2=None, K=5, alpha=0.8):
    a = 1.0
    sum = 0.0
    for k in range(K):
        xi1k = xi1[k]
        xi2k = xi2[k]
        temp = tf.pow(float(k+1), -1.0*alpha)*(xi1k*tf.sin(k*x)+xi2k*tf.cos(k*x))
        sum = sum + temp
    a = a+0.5*tf.sin(sum)
    u = tf.div(sum, a)
    return u


def random_equa2(xi1=0.25, xi2=-0.25, K=2, alpha=1.0):
    u = lambda x: tf.sin(xi1*tf.sin(np.pi*x)+0.5*xi2*tf.sin(2*np.pi*x))
    dux = lambda x: tf.cos(xi1*tf.sin(np.pi*x)+tf.pow(2, -1.0*alpha)*xi2*tf.sin(2*np.pi*x))*0.25*np.pi*(tf.cos(2*np.pi*x)-tf.cos(np.pi*x))
    f = lambda x: 1.0
    aeps = lambda x: 1.0
    return u, f, aeps


def random_sin_f(x=None, xi1=0.25, xi2=-0.25, K=2, alpha=1.0):
    f = tf.sin(xi1*tf.sin(np.pi*x)+0.5*xi2*tf.sin(2*np.pi*x))*np.pi*(xi2*tf.cos(2*np.pi*x)+xi1*tf.cos(np.pi*x))* \
        np.pi * (xi2*tf.cos(2*np.pi*x) + xi1*tf.cos(np.pi*x)) - \
        tf.cos(xi1*tf.sin(np.pi*x) + 0.5*xi2*tf.sin(2*np.pi*x)) * np.pi * \
        (-1.0*np.pi*xi1*tf.sin(np.pi*x) - 1.0*2.0* np.pi *xi2* tf.sin(2*np.pi*x))
    return f
