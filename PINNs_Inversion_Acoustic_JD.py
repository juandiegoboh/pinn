# %%
# Este código utiliza PINN de redes neuronales físicamente informadas (Raissi et al., 2019) para resolver el problema acústico inverso para una anomalía elipsodial de baja velocidad con una fuente puntual (Synthetic Crosswell). #Consulte el estudio de caso 3 del artículo Rasht-Behesht et al., 2021 para obtener una descripción completa de todos los parámetros involucrados

# %%
from SALib.sample import sobol_sequence
import os
import scipy.interpolate as interpolate
import timeit
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow.compat.v1 as tf
import pandas as pd

from _global import numero_experimento, path_folder

# %% Rutas

path_experimento = os.path.join(path_folder, f"experimentos\{numero_experimento}")
path_entrenamiento = os.path.join(path_folder, f"{path_experimento}/entrenamientoPINN")
path_parametros = os.path.join(path_experimento, "datos_propagacion.csv")

eventos_spec = os.path.join(path_folder, "event1")

sismos_spec = os.path.join(eventos_spec, "seismograms")
campos_spec = os.path.join(eventos_spec, "wavefields")

sismos_jd = os.path.join(path_experimento, "seismograms/")
campos_jd = os.path.join(path_experimento, "wavefields/")

# Lectura de parámetros de propagación
df = pd.read_csv(path_parametros)

#%% Ruta para guardar el resultado del entrenamiento

if not os.path.exists(path_entrenamiento):
    os.mkdir(path_entrenamiento)

# %%
tf.compat.v1.reset_default_graph()
tf.disable_eager_execution()

x = tf.placeholder(tf.float64, shape=(None, 1))
z = tf.placeholder(tf.float64, shape=(None, 1))
t = tf.placeholder(tf.float64, shape=(None, 1))

# %%
nx = df["Puntos X"][0]  # número de nodos a lo largo del eje x. utilizado aquí para eliminar las regiones absorbentes de specfem del dominio computacional de la PINN
nz = df["Puntos Z"][0]

n_abs = df["Nodos abs"][0]  # de nodos para absorber condiciones de frontera

n_absx = n_abs  # nodos del lado izquierdo
n_absz = n_abs  # nodos del lado inferior

ax_spec = df["Tamaño X"][0]/1000  # tamaño del dominio antes de eliminar las regiones absorbentes
az_spec = df["Tamaño Z"][0]/1000
xsf = 1.3  # x ubicación de todos los sismómetros en specfem

dx = ax_spec/nx
dz = az_spec/nz
rho = 1.0
# dimensión del dominio en la dirección x para entrenamiento de PINNs. Tenga en cuenta que
ax = xsf-n_absx*dx
# solo tenemos que eliminar el grosor del la frontera absorbente de la izquierda ya que #xsf es (debe ser) más pequeño que donde comienza la frontera absorbente del lado derecho
az = az_spec-n_absz*dz  # dimensión del dominio en la dirección z

t_m = 0.5  # tiempo total para el entrenamiento de la PDE.
t_st = 0.1  # aquí es cuando se toma el primer I.C de specfem
t_s = 0.5  # serie de tiempo total utilizada de los sismogramas

s_spec = 7.856742e-04  # paso de tiempo
t01 = df["t01"][0]  # disposición inicial en este momento de specfem
t02 = df["t02"][0]  # segunda disposición "inicial" en este momento desde specfem
t_la = df["t03"][0]  # datos de prueba para comparar specfem y PINN entrenados

n_event = 1  # numero de eventos sísmicos
n_seis = df["Numero sismometros"][0]  # número de sismómetros; si los eventos tienen diferentes números de sismómetros, se deben cambiar las líneas que contienen n_seis
z0_s = az  # z ubicación del primer sismómetro de SPECFEM en el marco de referencia de PINN. Aquí debe estar en km mientras que en SPECFEM está en metros. Tenga en cuenta que aquí asumimos que los sismómetros NO están todos en la superficie y están en una línea vertical con la misma x; el primer sismómetro está en la superficie y el siguiente va más profundo

zl_s = 0.06-n_absz*dz  # z ubicación del último sismómetro en profundidad. esto no tiene que ser cero y puede ser mayor, especialmente si tiene una frontera absorbente en la parte inferior, cámbielo según lo que haya usado de specfem

Lx = 3  # esto es para escalar la velocidad de la onda en el EDP a través de escalar la coordenada x
Lz = 3  # esto es para escalar la velocidad de la onda en el EDP a través de escalar la coordenada z

# %%
# Se define la velocidad verdadera del subsuelo

def g(x, z, a, b, c, d):
    return ((x-c)**2/a**2+(z-d)**2/b**2)


alpha_true = 3-0.25 * \
    (1+tf.tanh(100*(1-g(x*Lx, z*Lz, 0.18, 0.1, 1.0-n_absx*dx, 0.3-n_absz*dz))))

# %%
# normalización de la entrada a la NN
ub = np.array([ax/Lx, az/Lz, (t_m-t_st)]).reshape(-1, 1).T
# lo mismo para la NN inversa que estima la velocidad de onda
ub0 = np.array([ax/Lx, az/Lz]).reshape(-1, 1).T

# %%
def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float64), dtype=tf.float64)

# %%
# Definición de las dos redes neuronales

def neural_net(X, weights, biases):
    num_layers = len(weights) + 1
    H = 2*(X/ub)-1  # normalization map to [-1 1]
    for l in range(0, num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))

    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y


def neural_net0(X, weights, biases):
    num_layers = len(weights) + 1
    H = 2*(X/ub0)-1
    for l in range(0, num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))

    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y

# %%
# capas para la NN que aproxima el potencial acústico escalar (campo de presión)
layers = [3]+[30]*3+[1]

# %%
L = len(layers)
weights = [xavier_init([layers[l], layers[l+1]]) for l in range(0, L-1)]
biases = [tf.Variable(tf.zeros((1, layers[l+1]), dtype=tf.float64))
          for l in range(0, L-1)]
# num_epoch = 10000001
num_epoch = 100001

# %%
# capas para que la segunda NN que aproxima la velocidad de la onda (alpha)
layers0 = [2]+[20]*5+[1]

# %%
L0 = len(layers0)
weights0 = [xavier_init([layers0[l], layers0[l+1]]) for l in range(0, L0-1)]
biases0 = [tf.Variable(tf.zeros((1, layers0[l+1]), dtype=tf.float64))
           for l in range(0, L0-1)]

# %%
learning_rate = 1.e-4

alpha_star = tf.tanh(neural_net0(tf.concat((x, z), axis=1), weights0, biases0))

# %%
# Se elige la casilla dentro de la cual se está haciendo la inversión
# Se elimina la capa absorbente de z_st para hacerlo con referencia a la coordenada de PINN
z_st = 0.1-n_absz*dz
z_fi = 0.45-n_absz*dz
x_st = 0.7-n_absx*dx
x_fi = 1.25-n_absx*dx
lld = 1000
alpha_bound = 0.5*(1+tf.tanh(lld*(z-z_st/Lz)))*0.5*(1+tf.tanh(lld*(-z+z_fi/Lz)))*0.5*(1+tf.tanh(lld*(x-x_st/Lx))) * \
    0.5*(1+tf.tanh(lld*(-x+x_fi/Lx))
         )  # confinando la inversión a un límite y no a toda la región

alpha = 3+2*alpha_star*alpha_bound

# %%
# Red neuronal que calcula potencial de onda acústica escalar
phi = neural_net(tf.concat((x, z, t), axis=1), weights, biases)

# Operador Laplaciano
P = (1/Lx)**2*tf.gradients(tf.gradients(phi, x)
                           [0], x)[0] + (1/Lz)**2*tf.gradients(tf.gradients(phi, z)[0], z)[0]

eq = tf.gradients(tf.gradients(phi, t)[0], t)[0] - alpha**2*P  # Ecuación de onda escalar

# %% Calculo del campo de desplazamiento
ux = tf.gradients(phi, x)[0]    # u=grad(phi)
uz = tf.gradients(phi, z)[0]

# campo de velocidad, para el error en caso de que los datos de sismogramas esten en velocidad
Vel_x = tf.gradients(ux, t)[0]  
Vel_z = tf.gradients(uz, t)[0]

# %%
# Volumen del entrenamiento
batch_size = 40000
n_pde = batch_size*2000
print('batch_size', ':', batch_size)
X_pde = sobol_sequence.sample(n_pde+1, 3)[1:, :]
# X_pde = np.load('event1/X_pde.npy')
X_pde[:, 0] = X_pde[:, 0] * ax/Lx
X_pde[:, 1] = X_pde[:, 1] * az/Lz
X_pde[:, 2] = X_pde[:, 2] * (t_m-t_st)

# %%
# condiciones iniciales para todos los eventos
# coordenadas en las que se registra la salida del campo de onda en specfem. Es lo mismo para todas las ejecuciones con el mismo sistema de mallado en specfem
X0 = np.loadtxt(f'{campos_jd}xz_componentes_campos.txt')

# %%
# specfem funciona con unidades de metros, por lo que debemos convertirlos a Km.
X0 = X0/1
X0[:, 0:1] = X0[:, 0:1]/Lx  # escalar el dominio espacial
X0[:, 1:2] = X0[:, 1:2]/Lz  # escalar el dominio espacial
xz = np.concatenate((X0[:, 0:1], X0[:, 1:2]), axis=1)

#%%
fig = plt.figure()
plt.scatter(X0[:,0], X0[:,1], s=0.15)
plt.show()
plt.close(fig)

# %%
n_ini = 50

xx, zz = np.meshgrid(np.linspace(0, ax/Lx, n_ini),
                     np.linspace(0, az/Lz, n_ini))
xxzz = np.concatenate((xx.reshape((-1, 1)), zz.reshape((-1, 1))), axis=1)
X_init1 = np.concatenate((xx.reshape((-1, 1)), zz.reshape((-1, 1)), 0.0*np.ones((n_ini**2, 1),
                         dtype=np.float64)), axis=1)  # para hacer cumplir las condiciones de la primera I.C
X_init2 = np.concatenate((xx.reshape((-1, 1)), zz.reshape((-1, 1)), (t02-t01)*np.ones(
    (n_ini**2, 1), dtype=np.float64)), axis=1)  # para hacer cumplir las condiciones de la segunda I.C

# %%
# Interpolación de specfem da como resultado solo la parte no absorbente del dominio
xf = n_absx*dx  # inicio de la parte no absorbente del dominio en specfem
zf = n_absz*dz
xxs, zzs = np.meshgrid(np.linspace(xf/Lx, xsf/Lx, n_ini),
                       np.linspace(zf/Lz, az_spec/Lz, n_ini))
xxzzs = np.concatenate((xxs.reshape((-1, 1)), zzs.reshape((-1, 1))), axis=1)

u_scl = 1/1  # escalar los datos de salida para cubrir el intervalo [-1 1]

#%%
fig = plt.figure()
plt.scatter(xxs*Lx, zzs*Lz, s=0.3)
plt.show()
plt.close(fig)

# %%
# cargando los campos de ondas desde specfem
wfs = sorted(os.listdir(campos_jd))
U0 = [np.loadtxt(campos_jd+f) for f in wfs]

# %% Interpolación de los datos
U_ini1 = interpolate.griddata(xz, U0[0], xxzzs, fill_value=0.0)
U_ini1x = U_ini1[:, 0:1]/u_scl
U_ini1z = U_ini1[:, 1:2]/u_scl

U_ini2 = interpolate.griddata(xz, U0[1], xxzzs, fill_value=0.0)
U_ini2x = U_ini2[:, 0:1]/u_scl
U_ini2z = U_ini2[:, 1:2]/u_scl

U_spec = interpolate.griddata(xz, U0[2], xxzzs, fill_value=0.0)  # Datos de prueba
U_specx = U_spec[:, 0:1]/u_scl
U_specz = U_spec[:, 1:2]/u_scl

# %%
# los datos del primer evento se han subido encima y debajo
# se añaden el resto de los eventos n-1 (este for solamente actúa cuando hay más de una fuente)
# for ii in range(n_event-1):
#     wfs = sorted(os.listdir('event'+str(ii+2)+'/wavefields/.'))
#     U0 = [np.loadtxt('event'+str(ii+2)+'/wavefields/'+f) for f in wfs]

#     U_ini1 = interpolate.griddata(xz, U0[0], xxzzs, fill_value=0.0)
#     U_ini1x += U_ini1[:, 0:1]/u_scl
#     U_ini1z += U_ini1[:, 1:2]/u_scl

#     U_ini2 = interpolate.griddata(xz, U0[1], xxzzs, fill_value=0.0)
#     U_ini2x += U_ini2[:, 0:1]/u_scl
#     U_ini2z += U_ini2[:, 1:2]/u_scl

#     U_spec = interpolate.griddata(xz, U0[2], xxzzs, fill_value=0.0)
#     U_specx += U_spec[:, 0:1]/u_scl
#     U_specz += U_spec[:, 1:2]/u_scl
# U_ini=U_ini.reshape(-1,1)

# %%
# plots de entradas para la suma de los eventos

fig = plt.figure()
plt.contourf(xx*Lx, zz*Lz, np.sqrt(U_ini1x**2+U_ini1z **
             2).reshape(xx.shape), 100, cmap='jet')
plt.xlabel('x')
plt.ylabel('z')
plt.title(f'Scaled I.C total disp. t={t01:.3f}')
plt.colorbar()
plt.axis('scaled')
plt.savefig(f'{path_entrenamiento}/Ini_total_disp_spec_sumEvents.png', dpi=400)
plt.show()
plt.close(fig)

# %%
fig = plt.figure()
plt.contourf(xx*Lx, zz*Lz, np.sqrt(U_ini2x**2+U_ini2z **
             2).reshape(xx.shape), 100, cmap='jet')
plt.xlabel('x')
plt.ylabel('z')
plt.title(f'Scaled sec I.C total disp. t={t02:.3f}')
plt.colorbar()
plt.axis('scaled')
plt.savefig(f'{path_entrenamiento}/sec_wavefield_input_spec_sumEvents.png', dpi=400)
plt.show()
plt.close(fig)

# %%
fig = plt.figure()
plt.contourf(xx*Lx, zz*Lz, np.sqrt(U_specx**2+U_specz **
             2).reshape(xx.shape), 100, cmap='jet')
plt.xlabel('x')
plt.ylabel('z')
plt.title(f'Test data: Total displacement t={round(t_la-t01, 4)}')
plt.colorbar()
plt.axis('scaled')
plt.savefig(f'{path_entrenamiento}/total_disp_spec_testData_sumEvents.png', dpi=400)
plt.show()
plt.close(fig)
###############################################################

# %%
# ----componenete Z de los sismogramas
# sismogramas de entrada para el primer evento

sms = sorted(os.listdir(sismos_jd))
smsz = [f for f in sms if f[-6] == 'Z']  # Z cmp seismos
seismo_listz = [np.loadtxt(sismos_jd+f) for f in smsz]  # Z cmp seismos

# %%
# el tiempo de specfem no comienza desde cero para los sismos, por lo que se desplazan hacia adelante a cero
t_spec = -seismo_listz[0][0, 0]+seismo_listz[0][:, 0]
# aquí incluimos solo una parte de los sismogramas de specfem que están dentro del dominio de tiempo de entrenamiento de PINN que es [t_st t_m]
cut_u = t_spec > t_s
# Cortar los sismogramas solo después del momento en que se usa la primera instantánea de specfem para los PINN
cut_l = t_spec < t_st
# este es el índice del eje de tiempo en specfem después del cual t>t_m
l_su = len(cut_u)-sum(cut_u)
l_sl = sum(cut_l)

# %%
l_f = 2  # sismogramas de submuestreo de specfem
# submuestreo cada l_s pasos de tiempo de specfem en el intervalo de entrenamiento
index = np.arange(l_sl, l_su, l_f)
l_sub = len(index)
# eje de tiempo submuestreado de specfem para los sismogramas
t_spec_sub = t_spec[index].reshape((-1, 1))

# cambiar el eje del tiempo de nuevo a cero. la longitud de t_spec_sub debe ser igual a t_m-t_st
t_spec_sub = t_spec_sub-t_spec_sub[0]

# %%
for ii in range(len(seismo_listz)):
    seismo_listz[ii] = seismo_listz[ii][index]


Sz = seismo_listz[0][:, 1].reshape(-1, 1)
for ii in range(len(seismo_listz)-1):
    Sz = np.concatenate((Sz, seismo_listz[ii+1][:, 1].reshape(-1, 1)), axis=0)


# %%
#################################################################
# sismogramas de entrada para el resto de los eventos agregados al primer evento. Cuando hayan mas eventos
# for ii in range(n_event-1):
#     sms = sorted(os.listdir('event'+str(ii+2)+'/seismograms/.'))
#     smsz = [f for f in sms if f[-6] == 'Z']  # Z cmp seismos
#     seismo_listz = [np.loadtxt(
#         'event'+str(ii+2)+'/seismograms/'+f) for f in smsz]

#     for jj in range(len(seismo_listz)):
#         seismo_listz[jj] = seismo_listz[jj][index]

#     Sze = seismo_listz[0][:, 1].reshape(-1, 1)
#     for jj in range(len(seismo_listz)-1):
#         Sze = np.concatenate(
#             (Sze, seismo_listz[jj+1][:, 1].reshape(-1, 1)), axis=0)

#     Sz += Sze
###########################################################

# %%
Sz = Sz/u_scl  # escalar la suma de todas las entradas del sismograma


# X_S es la colección de entrenamiento de coordenadas de entrada en el espacio-tiempo para todos los sismogramas
X_S = np.empty([int(np.size(Sz)), 3])

# %%
d_s = np.abs((zl_s-z0_s))/(n_seis-1)  # distancia entre sismometros

for i in range(len(seismo_listz)):
    X_S[i*l_sub:(i+1)*l_sub, ] = np.concatenate((ax/Lx*np.ones((l_sub, 1), dtype=np.float64),
                                                 (z0_s-i*d_s)/Lz*np.ones((l_sub, 1),                                                                     
                                                                         dtype=np.float64),
                                                 t_spec_sub), axis=1)

# %%
# ---- componenete en X de los sismogramas
# sismogramas de entrada para la primera fuente


sms = sorted(os.listdir(sismos_jd))
smsx = [f for f in sms if f[-6] == 'X']  # X cmp sismos
seismo_listx = [np.loadtxt(sismos_jd+f)
                for f in smsx]  # X cmp sismos

# %%
for ii in range(len(seismo_listx)):
    seismo_listx[ii] = seismo_listx[ii][index]


Sx = seismo_listx[0][:, 1].reshape(-1, 1)
for ii in range(len(seismo_listx)-1):
    Sx = np.concatenate((Sx, seismo_listx[ii+1][:, 1].reshape(-1, 1)), axis=0)

# %%
#################################################################
# sismogramas de entrada para el resto de los eventos agregados al primer evento.
# Cuando hayan más sismos.

# for ii in range(n_event-1):
#     sms = sorted(os.listdir('event'+str(ii+2)+'/seismograms/.'))
#     smsx = [f for f in sms if f[-6] == 'X']  # X cmp seismos
#     seismo_listx = [np.loadtxt(
#         'event'+str(ii+2)+'/seismograms/'+f) for f in smsx]

#     for jj in range(len(seismo_listx)):
#         seismo_listx[jj] = seismo_listx[jj][index]

#     Sxe = seismo_listx[0][:, 1].reshape(-1, 1)
#     for jj in range(len(seismo_listx)-1):
#         Sxe = np.concatenate(
#             (Sxe, seismo_listx[jj+1][:, 1].reshape(-1, 1)), axis=0)

#     Sx += Sxe
###########################################################

# %%
Sx = Sx/u_scl  # escalar la suma de todas las entradas del sismograma

# BC: Tensión libre en la parte superior y sin condiciones de frontera para otros lados (absorbente)
bcxn = 100
bctn = 50
x_vec = np.random.rand(bcxn, 1)*ax/Lx
t_vec = np.random.rand(bctn, 1)*(t_m-t_st)
xxb, ttb = np.meshgrid(x_vec, t_vec)
X_BC_t = np.concatenate((xxb.reshape(
    (-1, 1)), az/Lz*np.ones((xxb.reshape((-1, 1)).shape[0], 1)), ttb.reshape((-1, 1))), axis=1)

# %%
N1 = batch_size
N2 = X_init1.shape[0]
N3 = X_init2.shape[0]
N4 = X_S.shape[0]

# %% 
XX = np.concatenate((X_pde[0:batch_size], X_init1,
                    X_init2, X_S, X_BC_t), axis=0)

# Diccionario para entrenamiento
feed_dict1 = {x: XX[:, 0:1], z: XX[:, 1:2], t: XX[:, 2:3]}

# Funciones de costo
loss_pde = tf.reduce_mean(tf.square(eq[:N1, 0:1]))

loss_init_disp1 = tf.constant(0.0, dtype=tf.float64)
loss_init_disp2 = tf.constant(0.0, dtype=tf.float64)

# %% Ecuaciones de costo, dependen de Ux y Uz.
loss_init_disp1 = tf.reduce_mean(tf.square(ux[N1:(N1+N2), 0:1]-U_ini1x)) \
    + tf.reduce_mean(tf.square(uz[N1:(N1+N2), 0:1]-U_ini1z))

loss_init_disp2 = tf.reduce_mean(tf.square(ux[(N1+N2):(N1+N2+N3), 0:1]-U_ini2x)) \
    + tf.reduce_mean(tf.square(uz[(N1+N2):(N1+N2+N3), 0:1]-U_ini2z))

loss_seism = tf.reduce_mean(tf.square(ux[(N1+N2+N3):(N1+N2+N3+N4), 0:1]-Sx)) \
    + tf.reduce_mean(tf.square(uz[(N1+N2+N3):(N1+N2+N3+N4), 0:1]-Sz))

loss_BC = tf.reduce_mean(tf.square(P[(N1+N2+N3+N4):, 0:1]))

# Función de costo completa
loss = 1e-1*loss_pde + loss_init_disp1 + \
    loss_init_disp2+loss_seism+1e-1*loss_BC

# %%
# Optimizador Adam, en lugar de lbfgs
optimizer_Adam = tf.train.AdamOptimizer(learning_rate)
train_op_Adam = optimizer_Adam.minimize(loss)

xx0, zz0 = xx.reshape((-1, 1)), zz.reshape((-1, 1))

# %%
# Evaluacion de PINNs en el tiempo=0
X_eval01 = np.concatenate((xx0, zz0, 0*np.ones((xx0.shape[0], 1))), axis=1)
# evaluación de PINN en el momento en que se proporciona la segunda entrada de specfem
X_eval02 = np.concatenate(
    (xx0, zz0, (t02-t01)*np.ones((xx0.shape[0], 1))), axis=1)
# evaluar los PINN en un momento posterior> 0
X_evalt = np.concatenate(
    (xx0, zz0, (t_la-t01)*np.ones((xx0.shape[0], 1))), axis=1)

# %%
# este diccionario es para evaluar la condición inicial recuperada de los PINN en nuevos puntos de prueba distintos a los utilizados para el entrenamiento
feed_dict01 = {x: X_eval01[:, 0:1], z: X_eval01[:, 1:2], t: X_eval01[:, 2:3]}
# este diccionario es para evaluar la condición inicial recuperada de los PINN en nuevos puntos de prueba distintos a los utilizados para el entrenamiento
feed_dict02 = {x: X_eval02[:, 0:1], z: X_eval02[:, 1:2], t: X_eval02[:, 2:3]}
# este diccionario es para evaluar los PINN en un momento posterior> 0
feed_dict2 = {x: X_evalt[:, 0:1], z: X_evalt[:, 1:2], t: X_evalt[:, 2:3]}
feed_dict_seism = {x: X_S[:, 0:1], z: X_S[:, 1:2], t: X_S[:, 2:3]}
i = int(-1)
loss_eval = np.zeros((1, 7))
loss_rec = np.empty((0, 7))

# %%
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # alpha toma dos variables pero feed_dict01 tiene tres entradas. pero está bien y no causará ningún problema
    alpha_true0 = sess.run([alpha_true], feed_dict=feed_dict01)

alpha_true0 = alpha_true0[0].reshape((xx.shape))

#%%
np.save('event1/modelo_vel',alpha_true0)

# %% Figura para el modelo de velocidad real utilizado
fig = plt.figure()
plt.contourf(Lx*xx, Lz*zz, alpha_true0.reshape((xx.shape)), 100, cmap='jet')
plt.xlabel('x')
plt.ylabel('z')
plt.title(r'True acoustic wavespeed ($\alpha$)')
plt.colorbar()
plt.axis('scaled')
plt.plot(Lx*0.99*X_S[:, 0], Lz*X_S[:, 1], 'r*', markersize=5)
plt.savefig(f'{path_entrenamiento}/True_wavespeed.png', dpi=400)
plt.show()
plt.close(fig)

# %%
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    alpha_plot = sess.run([alpha], feed_dict=feed_dict01)

alpha_plot = alpha_plot[0].reshape(xx.shape)

fig = plt.figure()
plt.contourf(xx*Lx, zz*Lz, alpha_plot.reshape((xx.shape)), 100, cmap='jet')
plt.xlabel('x')
plt.ylabel('z')
plt.title(r'Initial guess ($\alpha$)')
plt.colorbar()
plt.axis('scaled')
plt.savefig(f'{path_entrenamiento}/Ini_guess_wavespeed.png', dpi=400)
plt.show()
plt.close(fig)

# %% Entrenamiento de la red PINN

bbn = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    start = timeit.default_timer()
    for epoch in range(num_epoch):

        sess.run(train_op_Adam, feed_dict=feed_dict1)

        if epoch % 200 == 0:
            stop = timeit.default_timer()
            print('Time: ', stop - start)
            loss_val, loss_pde_val, loss_init_disp1_val, loss_init_disp2_val, loss_seism_val, loss_BC_val \
                = sess.run([loss, loss_pde, loss_init_disp1, loss_init_disp2, loss_seism, loss_BC], feed_dict=feed_dict1)

            print('Epoch: ', epoch, ', Loss: ', loss_val, ', Loss_pde: ',
                  loss_pde_val, ', Loss_init_disp1: ', loss_init_disp1_val)
            print(', Loss_init_disp2: ', loss_init_disp2_val,
                  'Loss_seism: ', loss_seism_val, 'Loss_stress: ', loss_BC_val)

            ux01 = sess.run([ux], feed_dict=feed_dict01)
            uz01 = sess.run([uz], feed_dict=feed_dict01)
            ux02 = sess.run([ux], feed_dict=feed_dict02)
            uz02 = sess.run([uz], feed_dict=feed_dict02)
            uxt = sess.run([ux], feed_dict=feed_dict2)
            uzt = sess.run([uz], feed_dict=feed_dict2)
            uz_seism_pred = sess.run([uz], feed_dict=feed_dict_seism)
            ux_seism_pred = sess.run([ux], feed_dict=feed_dict_seism)
            alpha0 = sess.run([alpha], feed_dict=feed_dict01)
            i = i+1
            loss_eval[0, 0], loss_eval[0, 1], loss_eval[0, 2], loss_eval[0, 3], loss_eval[0, 4], loss_eval[0, 5], loss_eval[0, 6]\
                = epoch, loss_val, loss_pde_val, loss_init_disp1_val, loss_init_disp2_val, loss_seism_val, loss_BC_val

            loss_rec = np.concatenate((loss_rec, loss_eval), axis=0)

            # Defining a new training batch for both PDE and B.C input data
            x_vec = np.random.rand(bcxn, 1)*ax/Lx
            t_vec = np.random.rand(bctn, 1)*(t_m-t_st)
            xxb, ttb = np.meshgrid(x_vec, t_vec)
            X_BC_t = np.concatenate((xxb.reshape(
                (-1, 1)), az/Lz*np.ones((xxb.reshape((-1, 1)).shape[0], 1)), ttb.reshape((-1, 1))), axis=1)

            bbn = bbn+1
            XX = np.concatenate(
                (X_pde[bbn*batch_size:(bbn+1)*batch_size], X_init1, X_init2, X_S, X_BC_t), axis=0)
            # This dictionary is for training
            feed_dict1 = {x: XX[:, 0:1], z: XX[:, 1:2], t: XX[:, 2:3]}

            U_PINN01 = ((ux01[0].reshape(xx.shape))**2 +
                        (uz01[0].reshape(xx.shape))**2)**0.5
            U_PINN02 = ((ux02[0].reshape(xx.shape))**2 +
                        (uz02[0].reshape(xx.shape))**2)**0.5
            U_PINNt = ((uxt[0].reshape(xx.shape))**2 +
                       (uzt[0].reshape(xx.shape))**2)**0.5
            U_diff = np.sqrt(U_specx**2+U_specz**2).reshape(xx.shape)-U_PINNt
            fig = plt.figure()
            plt.contourf(xx*Lx, zz*Lz, U_PINN01, 100, cmap='jet')
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title(r'PINNs $U(x,z,t=$'+str(0)+r'$)$')
            plt.colorbar()
            plt.axis('scaled')
            plt.savefig(f'{path_entrenamiento}/Total_Predicted_dispfield_t='+str(0)+'.png', dpi=400)
            # plt.show()
            plt.close(fig)
            fig = plt.figure()
            plt.contourf(xx*Lx, zz*Lz, U_PINN02, 100, cmap='jet')
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title(r'PINNs $U(x,z,t=$'+str(round(t02-t01, 4))+r'$)$')
            plt.colorbar()
            plt.axis('scaled')
            plt.savefig(f'{path_entrenamiento}/Total_Predicted_dispfield_t=' +
                        str(round(t02-t01, 4))+'.png', dpi=400)
            # plt.show()
            plt.close(fig)
            fig = plt.figure()
            plt.contourf(xx*Lx, zz*Lz, U_PINNt, 100, cmap='jet')
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title(r'PINNs $U(x,z,t=$'+str(round((t_la-t01), 4))+r'$)$')
            plt.colorbar()
            plt.axis('scaled')
            plt.savefig(f'{path_entrenamiento}/Total_Predicted_dispfield_t=' +
                        str(round((t_la-t01), 4))+'.png', dpi=400)
            # plt.show()
            plt.close(fig)

            fig = plt.figure()
            plt.contourf(xx*Lx, zz*Lz, U_diff, 100, cmap='jet')
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title(r'Total disp. Specfem-PINNs ($t=$' +
                      str(round((t_la-t01), 4))+r'$)$')
            plt.colorbar()
            plt.axis('scaled')
            plt.savefig(f'{path_entrenamiento}/Pointwise_Error_spec_minus_PINNs_t=' +
                        str(round((t_la-t01), 4))+'.png', dpi=400)
            # plt.show()
            plt.close(fig)

            fig = plt.figure()
            plt.contourf(
                xx*Lx, zz*Lz, alpha0[0].reshape(xx.shape), 100, cmap='jet')
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title(r'Inverted $\alpha$')
            plt.colorbar()
            plt.axis('scaled')
            plt.savefig(f'{path_entrenamiento}/inverted_alpha.png', dpi=400)
            # plt.show()
            plt.close(fig)

            fig = plt.figure()
            plt.contourf(xx*Lx, zz*Lx, alpha_true0 -
                         (alpha0[0].reshape(xx.shape)), 100, cmap='jet')
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title(r' $\alpha$ misfit (true-inverted)')
            plt.colorbar()
            plt.axis('scaled')
            plt.savefig(f'{path_entrenamiento}/alpha_misfit.png', dpi=400)
            # plt.show()
            plt.close(fig)

            fig = plt.figure()
            plt.plot(loss_rec[0:, 0], loss_rec[0:, 4], 'g', label='ini_disp2')
            plt.plot(loss_rec[0:, 0], loss_rec[0:, 6], 'black', label='B.C')
            plt.plot(loss_rec[0:, 0], loss_rec[0:, 1], '--y', label='Total')
            plt.plot(loss_rec[0:, 0], loss_rec[0:, 2], 'r', label='PDE')
            plt.plot(loss_rec[0:, 0], loss_rec[0:, 3], 'b', label='ini_disp1')
            plt.plot(loss_rec[0:, 0], loss_rec[0:, 5], 'c', label='Seism')
            plt.yscale("log")
            plt.xlabel('epoch')
            plt.ylabel('misfit')
            plt.legend()
            plt.savefig(f'{path_entrenamiento}/misfit.png', dpi=400)
            # plt.show()
            plt.close(fig)

            fig = plt.figure()
            plt.plot(X_S[600:750, 2], Sz[600:750],
                     'ok', mfc='none', label='Input')
            plt.plot(X_S[600:750, 2], uz_seism_pred[0]
                     [600:750], 'r', label='PINNs')
            plt.legend()
            plt.title(r' Vertical Seismogram z='+str(round(az-d_s, 4)))
            plt.savefig(f'{path_entrenamiento}/ZSeismograms_compare_z=' +
                        str(round(az-d_s, 4))+'.png', dpi=400)
            # plt.show()
            plt.close(fig)

            fig = plt.figure()
            plt.plot(X_S[600:750, 2], Sx[600:750],
                     'ok', mfc='none', label='Input')
            plt.plot(X_S[600:750, 2], ux_seism_pred[0]
                     [600:750], 'r', label='PINNs')
            plt.legend()
            plt.title(r' Horizontal Seismogram z='+str(round(az-d_s, 4)))
            plt.savefig(f'{path_entrenamiento}/XSeismograms_compare_z=' +
                        str(round(az-d_s, 4))+'.png', dpi=400)
            # plt.show()
            plt.close(fig)

            w_f = sess.run(weights)  # saving weights
            b_f = sess.run(biases)  # saving biases
            w_alph = sess.run(weights0)  # saving weights for the inverse NN
            b_alph = sess.run(biases0)
            with open(f'{path_entrenamiento}/recorded_weights.pickle', 'wb') as f:
                pickle.dump(['The first tensor contains weights, the second biases and the third losses',
                            w_f, b_f, w_alph, b_alph, loss_rec], f)
